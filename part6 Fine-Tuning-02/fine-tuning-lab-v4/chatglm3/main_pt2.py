import logging
import os
import sys
import torch
import json
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    set_seed, PreTrainedModel, AutoModelForCausalLM,
)
from trainer import PrefixTrainer
from arguments import ModelArguments, DataTrainingArguments
from data_preprocess import sanity_check, MultiTurnDataset

# 初始化日志记录
logger = logging.getLogger(__name__)


def setup_logger(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 配置huggingface的日志记录
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def load_model(model_args):
    print(f'------加载chatglm3-6b模型------')
    # 加载预训练的chatglm3-6b的model config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    print(f'------加载预训练的chatglm3-6b的tokenizer------')
    # 加载预训练的chatglm3-6b的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        config=config,
    )
    model = model.quantize(model_args.quantization_bit)
    # model = model.float()
    # model.transformer.prefix_encoder.float()
    print_model_size(model)
    # 判断是否加载pt2的checkpoint来继续训练
    # if model_args.ptuning_checkpoint is not None:
    #     print(f'------加载pt2的checkpoint来继续训练------')
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #     prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
    #     new_prefix_state_dict = {}
    #     for k, v in prefix_state_dict.items():
    #         if k.startswith("transformer.prefix_encoder."):
    #             new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    #     model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # else:  # 不加载pt2 checkpoint则直接加载model
    #     print(f'------不加载pt2 checkpoint则直接加载model------')
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    # # 如果有设置quantization则以int数值加载不参与更新的参数，用以节省显存
    # if model_args.quantization_bit is not None:
    #     print(f"-------Quantized to {model_args.quantization_bit} bit-------")
    #     model = model.quantize(model_args.quantization_bit)
    # # pt2训练，为要训练的prefix_encoder参数使用更高数值精度的float32
    # if model_args.pre_seq_len is not None:
    #     print(f'------pt2训练，为要训练的prefix_encoder参数使用更高数值精度的float32------')
    #     model = model.half()
    #     model.transformer.prefix_encoder.float()
    # # 全量参数finetune训练，本次实验中不会使用该模式，需要很高的显存配置
    # else:
    #     print(f'------全量参数finetune训练------')
    #     model = model.float()

    return tokenizer, model
def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")

def main():
    # 解析传入的命令行参数
    print(f'------开始解析参数------')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f'------解析参数完成------')
    # 初始化工作
    print(f'------初始化工作------')
    setup_logger(training_args)
    set_seed(training_args.seed)
    tokenizer, model = load_model(model_args)
    print(f'------初始化工作完成------')
    # 准备训练数据集并处理成所需格式
    print(f'------准备训练数据集并处理成所需格式------')
    if training_args.do_train:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]

        train_dataset = MultiTurnDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
        )

        # if training_args.local_rank < 1:
        #    sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)
    if training_args.do_eval:
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]

        eval_dataset = MultiTurnDataset(
            eval_data,
            tokenizer,
            data_args.max_seq_length,
        )
    print(f'------准备训练数据集并处理成所需格式完成------')
    # 将数据集中样本批处理成张量
    print(f'------将数据集中样本批处理成张量------')
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )
    print(f'------将数据集中样本批处理成张量完成------')
    # 配置trainer，相比base trainer重写了保存参数的功能
    print(f'------配置trainer------')
    trainer = PrefixTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        save_changed=model_args.pre_seq_len is not None
    )
    print(f'------配置trainer完成------')
    # 开始训练
    print(f'------开始训练------')
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        print(f'111111111')
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print(f'222222222')
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()
    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
