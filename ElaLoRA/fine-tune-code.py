from loralib.elalora import SVDLinear, RankAllocator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
import math


dataset = preprocess_dataset('C:/Users/Lenovo/Desktop/NLP/Final_project/IMDB Dataset.csv')

model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# SVDLinear  BERT/ModernBERT
#TARGET_LINEAR_TOKENS = {"q_proj","k_proj","v_proj","o_proj","dense","classifier"}
#TARGET_LINEAR_TOKENS = {"query","key","value","dense","intermediate","output","classifier"}
TARGET_LINEAR_TOKENS = {"Wqkv", "Wo", "Wi","dense","classifier"}

def replace_linear_with_svdlinear(module, prefix=""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(tok in full for tok in TARGET_LINEAR_TOKENS):
            in_f, out_f, has_bias = child.in_features, child.out_features, child.bias is not None
            setattr(module, name, SVDLinear(in_f, out_f, bias=has_bias))
        else:
            replace_linear_with_svdlinear(child, full)

replace_linear_with_svdlinear(base_model)

from loralib.elalora import SVDLinear
svd_names = [n for n, m in base_model.named_modules() if isinstance(m, SVDLinear)]
print("SVDLinear count =", len(svd_names))
print(svd_names[:20])

#  Freeze base / train เฉพาะ SVDLinear 
for p in base_model.parameters():
    p.requires_grad = False

for m in base_model.modules():
    if isinstance(m, SVDLinear):
        for p in m.parameters():
            p.requires_grad = True

if hasattr(base_model, "classifier"):
    for p in base_model.classifier.parameters():
        p.requires_grad = True

# เปิด classifier 
if hasattr(base_model, "classifier"):
    for p in base_model.classifier.parameters():
        p.requires_grad = True

# ✅ เพิ่ม: เปิดทุกชั้นที่ลงท้ายด้วย "norm"
for name, module in base_model.named_modules():
    if name.endswith("norm"):
        for p in module.parameters():
            p.requires_grad = True

# ✅ Optimizer: adapters กับ head
seen = set()

adap_params = []
for m in base_model.modules():
    if isinstance(m, SVDLinear):
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                adap_params.append(p)
                seen.add(id(p))

head_params = []
if hasattr(base_model, "classifier"):
    for p in base_model.classifier.parameters():
        if p.requires_grad and id(p) not in seen:
            head_params.append(p)
            seen.add(id(p))

param_groups = []
if adap_params:
    param_groups.append({"params": adap_params, "lr": 1.5e-3, "weight_decay": 0.01})
if head_params:
    param_groups.append({"params": head_params, "lr": 2.0e-3, "weight_decay": 0.0})

optimizer = AdamW(param_groups)

# ✅ Steps 
epochs = 5
'''
tmp_pipeline = FineTuningPipeline(
    dataset=dataset,
    tokenizer=tokenizer,
    model=base_model,
    optimizer=optimizer,
    val_size=0.1,
    epochs=epochs,
    seed=42,
    allocator=None  # ยังไม่ส่ง allocator เข้าไป
)
'''

#steps_per_epoch = len(tmp_pipeline.train_dataloader)
#estimated_total_steps = steps_per_epoch * epochs

allocator = RankAllocator(
    model=base_model,
    lora_r=24,
    target_rank=24,
    init_warmup=1,         # dummy
    final_warmup=1,        # dummy
    mask_interval=1,       # dummy
    total_step=1,
    #init_warmup=int(0.10 * estimated_total_steps),
    #final_warmup=int(0.60 * estimated_total_steps),
    #mask_interval=max(50, int(0.10 * estimated_total_steps)),
    #total_step=estimated_total_steps,
    beta1=0.85,
    beta2=0.85
)


loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
# ✅ Fine-tuning pipeline  use 320
fine_tuned_model = FineTuningPipeline(
    dataset=dataset,
    tokenizer=tokenizer,
    model=base_model,
    optimizer=optimizer,
    loss_function=loss_fn,   
    val_size=0.1,
    epochs=epochs,
    seed=42,
    allocator=allocator
)

### adjust code ver2
