import torch,json
from transformers import AutoModel, AutoModelForImageTextToText, Trainer,Seq2SeqTrainer, TrainingArguments, AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset,DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import snapshot_download #, AutoTokenizer
import argparse,random,os
import numpy as np
from tqdm import tqdm
# 或者更简洁的方式
import warnings
warnings.filterwarnings('ignore')

def dict_to_device(data, device):
    for k in data:
        if data[k] is None:
            continue
        if isinstance(data[k], dict):
            data[k] = dict_to_device(data[k], device)
        else:
            if hasattr(data[k],'to'):
                data[k] = data[k].to(device)

    return data
def convert_sharegpt_json_to_qwen3vl_format(lf_json, data_dir=None):
    """
    将sharegpt格式的单条JSON转换为API请求消息格式
    
    该函数处理包含多媒体内容（图像、视频）的对话数据，将特殊标记转换为实际的多媒体URL或base64编码
    
    Args:
        lf_json (dict): LF格式的输入JSON，包含对话消息和多媒体信息
            - messages: 对话消息列表，预期为[用户消息, 助手消息]
            - images: 图像文件路径列表（可选）
            - videos: 视频文件路径列表（可选）
        data_dir (str, optional): 多媒体文件的根目录。如果提供，会与相对路径拼接
    
    Returns:
        list: 包含单个用户消息的列表，消息内容为分段的多媒体和文本内容
    
    Raises:
        AssertionError: 如果输入格式不符合预期（必须是2条消息，且角色顺序为用户->助手）
    
    Example:
        输入LF格式:
        {
            "messages": [
                {"role": "user", "content": "请描述这个<image>和<video>"},
                {"role": "assistant", "content": "..."}
            ],
            "images": ["img1.jpg"],
            "videos": ["video1.mp4"]
        }
        
        输出API请求格式:
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这个"},
                    {"type": "image_url", "image_url": {"url": "path/to/img1.jpg"}},
                    {"type": "text", "text": "和"},
                    {"type": "video_url", "video_url": {"url": "path/to/video1.mp4"}}
                ]
            }
        ]
    """
    assert len(lf_json['messages'])==2 and lf_json['messages'][0]['role']=='user' and lf_json['messages'][1]['role']=='assistant'
    if 'images' in lf_json:
        images = lf_json['images']
    else:
        images = []
    if 'videos' in lf_json:
        videos = lf_json['videos']
    else:
        videos = []
    if data_dir is not None:
        images = [os.path.join(data_dir,i) for i in images]
        videos = [os.path.join(data_dir,i) for i in videos]
    messages = [{"role": "user", "content":[]}]
    content = lf_json['messages'][0]['content']
    while len(content)>0:
        image_pos = content.find("<image>")
        video_pos = content.find("<video>")
        if image_pos==-1: image_pos = np.inf
        if video_pos==-1: video_pos = np.inf

        if image_pos==0:
            messages[0]["content"].append( {"type": "image", 
                                            "image": images[0]} )
            images = images[1:]
            content = content[len("<image>"):]
        elif video_pos==0:
            messages[0]["content"].append( {"type": "video", 
                                            "video": videos[0]} )
            videos = videos[1:]
            content = content[len("<video>"):]
        else:
            cut_pos = min(image_pos, video_pos)
            if cut_pos==np.inf:
                messages[0]["content"].append( {"type": "text", 
                                                "text": content} )
                content = []
            else:
                messages[0]["content"].append( {"type": "text", 
                                                "text": content[:cut_pos]} )
                content = content[cut_pos:]
    messages.append({"role": "assistant", "content":[{"type": "text", "text": lf_json['messages'][1]['content']}]})
    return messages
def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
class Qwen3_VL_Lora_dataset(Dataset):
    def __init__(self, json_path, resource_path, tasklab2id=None,id2tasklab=None, ignore_keys=[]):
        self.json_path = json_path
        self.resource_path = resource_path
        with open(self.json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        # expand and json.loads all result json
        for i in range(len(data)):
            tmp = json.loads(data[i]['messages'][1]['content'])
            tmp = flatten_dict(tmp)
            data[i]['messages'][1]['content'] = tmp
        
        print("Preparing data messages, converting label to id and constructing the mapping dict...")
        messages,labels = [],[]
        tasks = [t for t in data[0]['messages'][1]['content'].keys() if t not in ignore_keys]
        print(f"Found {len(tasks)} classification tasks:", tasks)

        if tasklab2id is None or id2tasklab is None:
            tasklab2id,id2tasklab = set(),[]
            for idx,item in enumerate(tqdm(data)):
                tmp = item['messages'][1]['content']
                for t in tasks:
                    if t in ignore_keys: continue # 忽略某些key
                    tmp[t] = tmp[t].split(',')
                    for lab in tmp[t]:
                        if lab not in tasklab2id:
                            tasklab2id |= set({f"{t}-{lab}"})

                messages.append( convert_sharegpt_json_to_qwen3vl_format(data[idx], data_dir=self.resource_path) )
                labels.append(tmp)

            id2tasklab = sorted(list(tasklab2id))
            tasklab2id = {t:idx for idx,t in enumerate(id2tasklab)}
        else:
            print("Found existed label mapping dict, use it directly!!!")
            for idx,item in enumerate(tqdm(data)):
                tmp = item['messages'][1]['content']
                for t in tasks:
                    if t in ignore_keys: continue # 忽略某些key
                    tmp[t] = tmp[t].split(',')
                messages.append( convert_sharegpt_json_to_qwen3vl_format(data[idx], data_dir=self.resource_path) )
                labels.append(tmp)

        for i in range(len(labels)):
            item = labels[i]
            labels[i] = np.zeros(len(tasklab2id), dtype=np.int32)
            for k in item:
                if k in ignore_keys: continue # 忽略某些key
                tmp = [f"{k}-{v}" for v in item[k]]
                labels[i][ [tasklab2id[j] for j in tmp] ] = 1


        self.tasks = tasks
        self.messages = messages
        self.labels = labels
        self.id2tasklab = id2tasklab
        self.tasklab2id = tasklab2id
    
    def __len__(self):
        return len(self.messages)
    def __getitem__(self, index):
        return {"messages": self.messages[index][0], 
                "label": self.labels[index]}
class Qwen3_VL_Lora_Collator:
    def __init__(self, processor, num_of_pl_tokens=16, max_length=4096):
        self.processor = processor
        self.max_length = max_length
        self.num_of_pl_tokens = num_of_pl_tokens
    def __call__(self, data):
        assert len(data)==1 # 目前只支持batchsize=1

        inputs = self.processor.apply_chat_template(
            [data[0]['messages']],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        added_pl_tokens = [processor.tokenizer.get_vocab()['<unk>']]*self.num_of_pl_tokens
        inputs['input_ids'] = torch.cat([inputs['input_ids'][0],torch.tensor(added_pl_tokens, dtype=inputs['input_ids'].dtype)])[None]
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'][0],torch.tensor([1]*len(added_pl_tokens), dtype=inputs['attention_mask'].dtype)])[None]

        labels = torch.tensor(data[0]['label'], dtype=torch.long)[None]
        return {'inputs':inputs, 'labels':labels}

import torch
import torch.nn as nn
import torch.nn.functional as F
class LabelWiseAttention(nn.Module):
    def __init__(self, inSize, classNum):
        super(LabelWiseAttention, self).__init__()
        self.U = nn.Linear(inSize, classNum)
    def forward(self, X):
        # X: batchSize × seqLen × inSize
        alpha = F.softmax(self.U(X), dim=1) # => batchSize × seqLen × classNum
        X = torch.matmul(X.transpose(1,2), alpha) # => batchSize × inSize × classNum
        return X.transpose(1,2)
class HuggingfaceModelWithMTP(nn.Module):
    def __init__(self, model, pl_tkn_num, class_num):
        super(HuggingfaceModelWithMTP, self).__init__()
        hdn_size = model.config.text_config.hidden_size
        
        self.model = model
        self.label_proj = LabelWiseAttention(hdn_size, class_num)
        self.label_linear = nn.Linear(hdn_size, 1)
        
        self.value_proj = LabelWiseAttention(hdn_size, 1)
        self.value_linear = nn.Linear(hdn_size, 1)

        self.pl_tkn_num = pl_tkn_num
    def forward(self, inputs, **kwargs):
        tmp = self.model.model.model(**inputs).last_hidden_state
        lm_part = tmp[...,:-num_of_pl_tokens,:] # B,L1,D
        pl_part = tmp[...,-num_of_pl_tokens-1:,:] # B,L2,D

        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0] # B,1,D
        
        lab_part = self.label_proj(torch.cat([lm_part_pooled, pl_part],dim=1)) # B,C,D
        val_part = self.value_proj(torch.cat([lm_part_pooled, pl_part, lab_part],dim=1)) # B,1,D
        return {'y_logit':self.label_linear(lab_part)[...,0], 'v_head':self.value_linear(val_part)[...,0]}

from typing import TYPE_CHECKING, Any, Callable, Optional, Union
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在初始化时定义损失函数
        self.loss_label = nn.MultiLabelSoftMarginLoss()
        self.loss_value = nn.MSELoss()
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        loss1 = self.loss_label(outputs['y_logit'],labels)
        return (loss1, outputs) if return_outputs else loss1
        
        with torch.no_grad():
            acc = torch.mean(((outputs['y_logit'].detach()>0)==labels).to(outputs['y_logit'].dtype), dim=1, keepdims=True).detach().data # B,1
        loss2 = self.loss_value(outputs['v_head'], acc)

        total_loss = loss1+0.1*loss2
        
        return (total_loss, outputs) if return_outputs else total_loss

from sklearn import metrics as skmetrics
import scipy.stats as stats
def compute_metrics(eval_preds):
    logits,yt = eval_preds.predictions,eval_preds.label_ids
    # logits: N,C; N,1
    # yt: N,C
    yp = logits[0]>0
    vp = logits[1].reshape(-1) # N

    self_confidence = 1/(1+np.exp(logits[0])) # N,C
    acc_ij = yp==yt # N,C
    pcc_ij,spc_ij = stats.pearsonr(self_confidence.reshape(-1),acc_ij.reshape(-1))[0],stats.spearmanr(self_confidence.reshape(-1),acc_ij.reshape(-1))[0]
    
    acc_i = np.mean(yp==yt,axis=1,keepdims=True).reshape(-1) # N
    pcc_i,spc_i = stats.pearsonr(vp,acc_i)[0],stats.spearmanr(vp,acc_i)[0]
    
    acc = np.mean(acc_i)
    mif = skmetrics.f1_score(yp,yt, average='micro')
    precision = skmetrics.precision_score(yp,yt, average='samples')
    recall = skmetrics.recall_score(yp,yt, average='samples')
    f1 = skmetrics.f1_score(yp,yt, average='samples')

    # print(logits[0].shape, yt.shape)
    # print(logits[1].shape, acc_i.shape)
    eval_label_loss = F.multilabel_soft_margin_loss(torch.from_numpy(logits[0]),torch.from_numpy(yt))
    eval_value_loss = F.mse_loss(torch.from_numpy(logits[1].reshape(-1)),torch.from_numpy(acc_i.reshape(-1)))
    return {"label_loss":eval_label_loss,
            "value_loss":eval_value_loss,
            
            'ACC':acc, 'MiF':mif, 
            'sample-avg precision':precision,
            'sample-avg recall':recall, 
            'sample-avg f1':f1,

            'self-conf PCC': pcc_ij,
            'self-conf SPC': spc_ij,
            
            'value PCC': pcc_i,
            'value SPC': spc_i}

def unflatten_dict(flat_dict, sep='-'):
    result = {}    
    for flat_key, value in flat_dict.items():
        # 分割键为各级键名
        keys = flat_key.split(sep)
        # 当前处理的字典层级
        current_dict = result
        # 遍历所有键，除了最后一个
        for i, key in enumerate(keys[:-1]):
            # 如果键不存在，创建新字典
            if key not in current_dict:
                current_dict[key] = {}
            # 如果键存在但不是字典，说明有冲突，无法恢复
            elif not isinstance(current_dict[key], dict):
                raise ValueError(f"冲突: 键 '{key}' 在扁平化字典中同时是叶节点和中间节点")
            # 进入下一层
            current_dict = current_dict[key]
        # 设置最终的值
        last_key = keys[-1]
        # 检查最后一个键是否已存在（作为中间节点）
        if last_key in current_dict and isinstance(current_dict[last_key], dict):
            raise ValueError(f"冲突: 键 '{last_key}' 在扁平化字典中同时是叶节点和中间节点")
        current_dict[last_key] = value
    return result
def kv_list_to_dict(kv_list, sep=','):
    result = {}
    for key, value in kv_list:
        result.setdefault(key, [])
        result[key].append(value)
    if sep is None:
        return result
    return {k:sep.join(result[k]) for k in result}
def convert_ids_to_lab_json(id2tasklab, ids):
    tmp = np.array(id2tasklab)[ids]
    tmp = [i.split('-') for i in tmp]
    tmp = [["-".join(i[:-1]),i[-1]] for i in tmp]
    tmp = kv_list_to_dict(tmp)
    return unflatten_dict(tmp)

parser = argparse.ArgumentParser()
# data argument
parser.add_argument("--json_path", type=str)
parser.add_argument("--eval_json_path", type=str)
parser.add_argument("--resource_path", type=str)
parser.add_argument("--outdir", type=str, default='./')
# model argument
parser.add_argument("--path_to_ckt", type=str)
parser.add_argument("--model_name", type=str, default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument("--model_path", default=None)
parser.add_argument("--num_of_pl_tokens", type=int, default=16)
# training argument
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--max_length", type=int, default=65536)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--save_steps", type=int, default=128)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--num_train_epochs", type=int, default=16)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--deepspeed", type=str, default="")
parser.add_argument('--local_rank', type=int, default=0) # deepspeed需要该参数

# deepspeed SFT_main.py --json_path /app/data/test-wuyifan/MyProject/video_understanding_v2/train_samples_20251224/20251224141718/20251224141718-json/20251224141718-json-meixue-v2-clean-train.json --eval_json_path /app/data/test-wuyifan/MyProject/video_understanding_v2/train_samples_20251224/20251224141718/20251224141718-json/20251224141718-json-meixue-v2-clean-test.json --resource_path /app/data/test-wuyifan/MyProject/video_understanding_v2/train_samples_20251224 --gradient_accumulation_steps 2 --deepspeed "./ds_z2_config.json"

if __name__=='__main__':
    args = parser.parse_args()
    training_args = TrainingArguments(
            output_dir="./saves",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            logging_steps=int(args.logging_steps),
            save_steps=int(args.save_steps), # 每更新多少步保存一次模型
            save_total_limit=4,
            bf16=True,
            learning_rate=float(args.learning_rate),
            weight_decay=1e-2,
            warmup_ratio=float(args.warmup_ratio),
            deepspeed=args.deepspeed if len(args.deepspeed)>0 else None, 
            remove_unused_columns=False,
            label_names=["labels"],

            # gradient_checkpointing=True,
            dataloader_num_workers=8,
            # dataloader_persistent_workers=False,
            do_eval=True,
            eval_accumulation_steps=1,
            per_device_eval_batch_size=1,
            eval_strategy='steps',
            eval_steps=int(args.logging_steps),
            save_safetensors=False 
        )

    model_path = args.model_name
    num_of_pl_tokens = int(args.num_of_pl_tokens)
    
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, # device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_site = 'left'
    processor.tokenizer.truncation_side = 'left'

    # budget for image processor, since the compression ratio is 32 for Qwen3-VL, we can set the number of visual tokens of a single image to 256-1280 (32× spatial compression)
    # processor.image_processor.size = {"longest_edge": 1280*32*32, "shortest_edge": 256*32*32}
    processor.image_processor.size = {"longest_edge": 320*32*32, "shortest_edge": 64*32*32}

    # budget for video processor, we can set the number of visual tokens of a single video to 256-16384 (32× spatial compression + 2× temporal compression)
    # processor.video_processor.size = {"longest_edge": 16384*32*32*2, "shortest_edge": 256*32*32*2}
    processor.video_processor.size = {"longest_edge": 4096*32*32*2, "shortest_edge": 64*32*32*2}

    trainDS = Qwen3_VL_Lora_dataset(json_path=args.json_path,
                                    resource_path=args.resource_path, ignore_keys=['灯光与色彩-备注','美学亮点'])

    validDS = Qwen3_VL_Lora_dataset(json_path=args.eval_json_path,
                                    resource_path=args.resource_path, ignore_keys=['灯光与色彩-备注','美学亮点'],
                                    tasklab2id=trainDS.tasklab2id,id2tasklab=trainDS.id2tasklab)

    assert len(trainDS.id2tasklab)==len(validDS.id2tasklab)

    collater = Qwen3_VL_Lora_Collator(processor=processor, num_of_pl_tokens=num_of_pl_tokens, max_length=args.max_length)

    from peft import LoraConfig, TaskType, get_peft_model
    # apply LoRA
    lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            inference_mode=False, # 训练模式
            r = int(args.lora_rank), # Lora秩, 
            lora_alpha = int(args.lora_alpha), # Lora alpha
            lora_dropout = 0.0, # Dropout比例
        )
    base_model = get_peft_model(base_model, lora_config).to("cuda")
    base_model.print_trainable_parameters()

    model = HuggingfaceModelWithMTP(base_model, pl_tkn_num=num_of_pl_tokens, class_num=len(trainDS.tasklab2id))
    model = model.cuda()
    model.load_state_dict(torch.load(args.path_to_ckt))
    model.eval()

    trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=trainDS,
            eval_dataset=validDS,
            data_collator=collater,
            compute_metrics=compute_metrics,
            # tokenizer=tokenizer,
            # compute_loss_func=...,
            # compute_metrics=...,
        )
    
    with torch.no_grad():
        predictions = trainer.predict(validDS)
    print("========================================")
    print("================METRICS=================")
    print("========================================")
    print(predictions[-1])
    print("========================================")

    print(f"将预测结果写入{args.outdir}/generated_predictions.jsonl...")
    # 修正默认阈值，保证每个类别下必定有一个正标签
    def fix_thrshold_for_minimal_conf(id2tasklab, yp):
        tmp = [i.split('-') for i in id2tasklab]
        tmp = ["-".join(i[:-1]) for i in tmp]
        lab_conf = kv_list_to_dict(zip(tmp,yp), sep=None)
        _ = [[len(lab_conf[k]),np.max(lab_conf[k])] for k in lab_conf]
        
        thr = np.zeros(len(id2tasklab))
        assert sum([i[0] for i in _])==len(thr)
        sIdx = 0
        for item in _:
            eIdx = sIdx+item[0]
            thr[sIdx:eIdx] = min(0,item[1])
            sIdx = eIdx
        return thr
    jsonlines = []
    for x,yp,yt in zip(validDS,predictions.predictions[0],predictions.label_ids):
        thr = fix_thrshold_for_minimal_conf(trainDS.id2tasklab, yp)

        yp = convert_ids_to_lab_json(trainDS.id2tasklab, yp>=thr)
        yt = convert_ids_to_lab_json(trainDS.id2tasklab, yt==1)
        jsonlines.append( {"input":x['messages'], "predict":json.dumps(yp,ensure_ascii=False), "label":json.dumps(yt,ensure_ascii=False)} )
    with open(os.path.join(args.outdir, "./generated_predictions.jsonl"), 'w', encoding='utf8') as f:
        for jl in jsonlines:
            f.write(json.dumps(jl, ensure_ascii=False) + '\n')

