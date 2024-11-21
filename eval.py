import argparse
import json
import multiprocessing
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import matplotlib.pyplot as plt
import MeCab
import numpy as np
import onnxruntime as ort
import pandas as pd
import sacrebleu
import torch
from comet.models import load_from_checkpoint
from loguru import logger
from nltk.translate.bleu_score import corpus_bleu
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pecab import PeCab
from pythainlp.tokenize import word_tokenize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MarianTokenizer

from dataset.mt_dataset import StandardMTDataset
from models.get_model import get_model
from tools.gen_table import save_score_table

class LanguageTokenizer:
    def __init__(self):
        self.pecab = PeCab()
        self.tokenize_languages = {"zh-cn", "ja", "ko", "th"}
        self.mecab_tagger = None

    def get_mecab_tagger(self):
        if self.mecab_tagger is None:
            self.mecab_tagger = MeCab.Tagger("-Owakati")
        return self.mecab_tagger

    @staticmethod
    def remove_language_id(sentence: str) -> str:
        return re.sub(r">>.+<<", "", sentence)

    def tokenize(self, sentence: str, lang: str, is_multilingual: bool = False) -> str:
        if is_multilingual:
            sentence = self.remove_language_id(sentence)

        if lang not in self.tokenize_languages:
            return sentence

        sentence = re.sub(r"[^\w\s]", "", sentence)
        sentence = sentence.replace(" ", "")
        if lang == "ja":
            return self.get_mecab_tagger().parse(sentence).strip()
        if lang == "zh-cn":
            tmp = jieba.cut(sentence)
            return " ".join(tmp).strip()
        if lang == "ko":
            tmp = self.pecab.morphs(sentence)
            return " ".join(tmp).strip()
        if lang == "th":
            tmp = word_tokenize(sentence, engine="newmm")
            return " ".join(tmp).strip()
        return sentence


class Evaluation:
    def __init__(self, args: argparse.Namespace):
        self.eval_dataset = "flores101-dev"
        
        self.save_dir = Path(args.save_dir)
        self.model_dir = Path(args.model)
        self.comet_dir = Path(args.comet_dir)
        self.eval_data = Path(args.data)
        
        self.batch_size = args.batch_size
        self.num_workers = 0
        self.is_multilingual = args.is_multilingual
        self.use_multi_decoder = args.use_multi_decoder
        self.only_eval = args.only_eval
        self.only_infer = False

        self.results = None
        self.lang_pairs = None
        self.try_load_args_from_train_info()

        self.comet_model = self.get_comet_model(comet_dir=args.comet_dir)
        self.language_tokenier = LanguageTokenizer()

        self.infer_model = None
        if not self.only_eval:
            self.infer_model = self.get_infer_model(args)

        self.logger_infos()

    def try_load_args_from_train_info(self) -> None:
        train_info_path = self.model_dir / "train_info.json"
        
        if not train_info_path.exists():
            logger.warning(f"Can't find train_info.json in {train_info_path}. ")
            return

        with open(train_info_path, "r", encoding="utf-8") as f:
            train_info = json.load(f)

        # Check whether cli_args in train_info.json
        if not train_info.get("cli_args"):
            logger.warning(
                "<cli_args> not in train_info.json, check train_info.json version."
            )
            return

        logger.info("Loading train_info.json, overwritting params.")
        self.is_multilingual = train_info.get("cli_args").get("is_multilingual")
        self.use_multi_decoder = train_info.get("cli_args").get("use_multi_decoder")
        self.lang_pairs = train_info.get("lang_pairs")

    def logger_infos(self):
        if self.only_eval:
            logger.info("Only evaluate on local files.")

    def get_comet_model(self, comet_dir: str):
        checkpoint = os.path.join(comet_dir, "checkpoints/model.ckpt")
        comet_model = load_from_checkpoint(checkpoint_path=checkpoint).cuda()
        logger.info(f"Comet model: {self.comet_dir}")
        return comet_model

    def get_infer_model(self, args: argparse.Namespace):
        if not args.onnx:
            model, _ = get_model(
                model_dir=self.model_dir,
                train_data=None,
                val_data=self.eval_data,
                use_multi_decoder=self.use_multi_decoder,
            )
            if self.use_multi_decoder:
                self.is_multilingual = True
            return model.cuda()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 16
        session_options.inter_op_num_threads = 16

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        decoder_name = (
            "decoder_model_merged_int8.onnx"
            if args.use_int8
            else "decoder_model_merged.onnx"
        )
        return ORTModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            encoder_file_name="encoder_model_pre.onnx",
            decoder_file_name=decoder_name,
            use_cache=False,
            session_options=session_options,
            providers=providers,
        )

    def smooth(self, data: np.array, window_size: int = 100) -> np.array:
        """
        Apply moving average smoothing to the input data.

        Parameters:
        data (np.array): Input NumPy array.
        window_size (int): Window size for moving average, default is 100.

        Returns:
        np.array: Smoothed data.
        """
        series = pd.Series(data)
        moving_average = series.rolling(window=window_size).mean()
        return moving_average.to_numpy()

    def draw_eval_data(
        self,
        data: List[List[float]],
        titles: List[str],
        window_sizes: List[int],
        save_dir: Path,
    ) -> None:
        """
        Plot and save evaluation data.

        Parameters:
        data (List[List[float]]): List of data series to plot.
        titles (List[str]): Titles for each data series.
        window_sizes (List[int]): Window sizes for smoothing.
        save_dir (str): Directory to save the plot.

        Returns:
        None
        """
        assert len(data) == len(titles)
        num_plots = len(data)
        fig, axs = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
        if num_plots == 1:
            axs = [axs]

        for ax, metric, title in zip(axs, data, titles):
            ax.plot(metric, label="Original")
            for window_size in window_sizes:
                smoothed_metric = self.smooth(metric, window_size)
                ax.plot(smoothed_metric, label=f"Smoothed (window={window_size})")
            ax.set_title(title)
            ax.legend()

        plt.subplots_adjust(
            hspace=0.5, wspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95
        )
        plt.savefig(save_dir / "eval.png")

    def sort_results_by_bleu(self) -> None:
        results = self.results
        if results["error_spans"] is None:
            results["error_spans"] = ["None" for _ in range(len(results["bleus"]))]
        sorted_results = sorted(
            zip(
                results["translations"],
                results["labels"],
                results["origins"],
                results["bleus"],
                results["comets"],
                results["error_spans"],
            ),
            key=lambda x: x[3],
        )

        # 解压排序后的结果
        translations, labels, origins, bleus, comets, error_spans = zip(*sorted_results)

        # 更新字典中的列表
        results["translations"] = list(translations)
        results["labels"] = list(labels)
        results["origins"] = list(origins)
        results["bleus"] = list(bleus)
        results["comets"] = list(comets)
        results["error_spans"] = list(error_spans)

        self.results = results

    def save_sentence_score(self, save_dir: Path) -> None:
        file_path = save_dir / "sentence_score"
        with open(file_path, "w", encoding="utf-8") as fout:
            for i, bleu_score in enumerate(self.results["bleus"]):
                origin = self.results["origins"][i]
                translation = self.results["translations"][i]
                label = self.results["labels"][i]
                comet_score = self.results["comets"][i] * 100
                error_span = self.results["error_spans"][i]
                fout.write(
                    textwrap.dedent(f"""
                        ****
                        BLEU:        {bleu_score:.4f}
                        COMET:       {comet_score:.4f}
                        Source:      {origin}
                        Translation: {translation}
                        Reference:   {label}
                        ErrorSpan:   {error_span}
                    """)
                )

    def save_system_score(self, comet_system_score: float, save_dir: Path) -> None:
        references = [[label] for label in self.results["labels"]]
        all_bleu = corpus_bleu(references, self.results["translations"])

        file_path = save_dir / "corpus_score"

        with open(file_path, "w", encoding="utf-8") as fout:
            fout.write(f"BLEU:  {all_bleu * 100:.4f} \n")
            fout.write(f"COMET: {comet_system_score:.4f}")

            logger.info(f"BLEU:  {all_bleu * 100:.4f}")
            logger.info(f"COMET: {comet_system_score:.4f}")

            logger.info(f"Score folder: {save_dir}")

    def read_eval_file(self, save_dir: Path) -> Tuple[Dict, List]:
        self.results = {
            "labels": [],
            "translations": [],
            "origins": [],
            "bleus": [],
        }
        logger.info("Reading local eval files.")
        comet_data = []
        with open(
            save_dir / "ref", "r", encoding="utf-8"
        ) as ref_file, open(
            save_dir / "tra", "r", encoding="utf-8"
        ) as tra_file, open(
            save_dir / "orig", "r", encoding="utf-8"
        ) as orig_file:
            for label, translation, origin in zip(ref_file, tra_file, orig_file):
                label, translation, origin = (
                    label.strip(),
                    translation.strip(),
                    origin.strip(),
                )
                bleu = sacrebleu.sentence_bleu(translation, [label])

                self.results["translations"].append(translation)
                self.results["labels"].append(label)
                self.results["origins"].append(origin)
                self.results["bleus"].append(bleu.score)

                comet_data.append({"src": origin, "mt": translation, "ref": label})

        return comet_data

    def comet_prediction(self, data: List[Dict[str, str]]) -> float:
        outputs = self.comet_model.predict(
            data, batch_size=32, gpus=1, num_workers=self.num_workers
        )

        error_spans = getattr(getattr(outputs, "metadata", None), "error_spans", None)

        self.results["comets"] = outputs["scores"]
        self.results["error_spans"] = error_spans

        return outputs["system_score"]

    def infer_save(self, eval_data: Path, save_dir: Path) -> None:
        tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
        save_dir = save_dir

        eval_datasets = StandardMTDataset(
            eval_data, tokenizer, is_multilingual=self.is_multilingual, augment=False
        )

        dataloader = DataLoader(
            dataset=eval_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        with open(eval_data, "r") as fin:
            f_json = json.load(fin)
            tgt_lang = list(f_json["lang_pairs"].values())[0][0]

        os.makedirs(save_dir, exist_ok=True)
        with open(
            save_dir / "ref", "w", encoding="utf-8"
        ) as ref_file, open(
            save_dir / "tra", "w", encoding="utf-8"
        ) as tra_file, open(
            save_dir / "orig", "w", encoding="utf-8"
        ) as orig_file:
            for batch in tqdm(dataloader):
                inputs = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                outputs = self.infer_model.generate(inputs, num_beams=1)
                translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                labels_decoded = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                inputs_decoded = tokenizer.batch_decode(
                    inputs, skip_special_tokens=True
                )
                for translation, label, origin in zip(
                    translations, labels_decoded, inputs_decoded
                ):
                    origin = self.language_tokenier.remove_language_id(origin)
                    label = self.language_tokenier.tokenize(
                        label, tgt_lang, self.is_multilingual
                    )
                    translation = self.language_tokenier.tokenize(
                        translation, tgt_lang, self.is_multilingual
                    )
                    tra_file.write(translation + "\n")
                    ref_file.write(label + "\n")
                    orig_file.write(origin + "\n")

    def eval_with_metric(self, save_dir: Path) -> None:
        comet_data = self.read_eval_file(save_dir=save_dir)
        comet_system_score = self.comet_prediction(data=comet_data)

        self.sort_results_by_bleu()

        self.draw_eval_data(
            data=[self.results["bleus"], self.results["comets"]],
            titles=["BLEU", "COMET"],
            window_sizes=[10, 50, 100],
            save_dir=save_dir,
        )

        self.save_sentence_score(save_dir=save_dir)
        self.save_system_score(comet_system_score=comet_system_score, save_dir=save_dir)
    
    
    def get_multi_mode_folder(self):
        eval_data_dir = self.eval_data.parent
        model_name = self.model_dir.name
        score_dir = self.save_dir.parent
        
        mode = "multilingual"
        if self.use_multi_decoder:
            mode = "multi_deocder"
        
        # main folder (model_name)
        multilingual_save_dir = score_dir / f"{mode}-{model_name}"
        os.makedirs(multilingual_save_dir, exist_ok=True)
        return eval_data_dir, multilingual_save_dir
        
        
    def multi_mode_generator(self):
        eval_data_dir, multilingual_save_dir = self.get_multi_mode_folder()
        for src_lang, tgt_langs in self.lang_pairs.items():
            for tgt_lang in tgt_langs:
                _src_lang = src_lang.replace("zh-cn", "zh")
                _tgt_lang = tgt_lang.replace("zh-cn", "zh")
                eval_name = f"{self.eval_dataset}-{_src_lang}-{_tgt_lang}"
                
                eval_data = eval_data_dir / f"{eval_name}.json"
                save_dir = multilingual_save_dir / eval_name
                yield src_lang, tgt_lang, eval_data, save_dir


    def process_evaluation(self, eval_data: Path, save_dir: Path) -> None:
        if self.only_eval and self.only_infer:
            logger.error("<only_eval> and <only_infer> cannot both be True.")
            return
        if self.only_infer:
            self.infer_save(eval_data=eval_data, save_dir=save_dir)
            return
        if self.only_eval:
            self.eval_with_metric(save_dir=save_dir)
            return
        self.infer_save(eval_data=eval_data, save_dir=save_dir)
        self.eval_with_metric(save_dir=save_dir)
    

    def evaluate(
        self,
    ) -> None:
        if self.use_multi_decoder or self.is_multilingual:
            if self.use_multi_decoder:
                logger.info("Evaluation mode: multi-decoder")
            else:
                logger.info("Evaluation mode: multilingual")
                
            save_dir = None
            for src_lang, tgt_lang, eval_data, save_dir in self.multi_mode_generator():
                if self.use_multi_decoder:
                    self.infer_model.model.infer_lang = tgt_lang
                    logger.info(f"Set inference language: {tgt_lang}")  
                self.process_evaluation(
                    eval_data=eval_data,
                    save_dir=save_dir
                )
                
            eval_data_dir, multilingual_save_dir = self.get_multi_mode_folder()
            save_score_table(
                model_name=multilingual_save_dir.name,
                save_dir=multilingual_save_dir)
        else:
            logger.info("Evaluation mode: origin")
            self.process_evaluation(eval_data=self.eval_data, save_dir=self.save_dir)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--comet_dir", type=str)
    parser.add_argument("--save_dir", type=str, default="eval/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--use_int8", action="store_true")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--is_multilingual", action="store_true")
    parser.add_argument("--use_multi_decoder", action="store_true")
    args = parser.parse_args()

    evaluation = Evaluation(args)
    evaluation.evaluate()
