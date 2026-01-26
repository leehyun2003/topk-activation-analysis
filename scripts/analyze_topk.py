import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter
import os
import matplotlib.pyplot as plt
import re

# 벤다이어그램 라이브러리 체크
try:
    from matplotlib_venn import venn3
    has_venn = True
except ImportError:
    has_venn = False

# 1. 설정
MODEL_NAME = "Qwen/Qwen2.5-14B" 
TOP_K = 50
REP_COUNT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} with 4-bit quantization...")

# 메모리 절약을 위한 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. 모델 레이어 수 자동 감지 및 샘플링 구간 설정
num_layers = len(model.model.layers)
# 14B 모델은 레이어가 많으므로(예: 48개) 좀 더 세밀하게 7개 지점을 봅니다.
TARGET_LAYERS = [num_layers // 7 * i for i in range(1, 7)] + [num_layers - 1]
print(f"Detected {num_layers} layers. Analyzing: {TARGET_LAYERS}")

layer_activations = {l_idx: [] for l_idx in TARGET_LAYERS}

def get_swiglu_hook(l_idx):
    def hook(module, input, output):
        # SwiGLU 출력값 h 가로채기
        h = input[0].detach()
        _, topk_indices = torch.topk(h, k=TOP_K, dim=-1)
        layer_activations[l_idx].append(topk_indices.cpu())
    return hook

# Hook 등록
for l_idx in TARGET_LAYERS:
    model.model.layers[l_idx].mlp.down_proj.register_forward_hook(get_swiglu_hook(l_idx))

# 3. 데이터 추출 함수 (한 줄 텍스트 분할 처리)
def get_layerwise_fingerprints(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    for l_idx in TARGET_LAYERS: layer_activations[l_idx] = []
    counters = {l_idx: Counter() for l_idx in TARGET_LAYERS}

    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read().replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    print(f"Processing {len(sentences)} units from {file_path}...")
    
    for sent in sentences[:100]: 
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        
        for l_idx in TARGET_LAYERS:
            if layer_activations[l_idx]:
                # 마지막 토큰 혹은 전체 평균 활성화 중 선택 (여기서는 마지막 배치 처리)
                batch_indices = layer_activations[l_idx].pop().view(-1).tolist()
                counters[l_idx].update(batch_indices)

    return {l_idx: set([n for n, c in cnt.most_common(REP_COUNT)]) for l_idx, cnt in counters.items()}

# 4. 분석 실행
print("\n--- Starting Domain Analysis ---")
fp_med = get_layerwise_fingerprints("data/medical.txt")
fp_law = get_layerwise_fingerprints("data/law.txt")
fp_fusion = get_layerwise_fingerprints("data/medical_law.txt")

# 5. 시각화 및 결과 저장
def plot_venn_diagram(l_idx, med, law, fusion):
    if not has_venn: return
    plt.figure(figsize=(8, 8))
    venn3([med, law, fusion], set_labels=('Medical', 'Legal', 'Medical-Law'))
    plt.title(f"Node Activation Venn Diagram at Layer {l_idx} (14B)")
    plt.savefig(f"venn_14b_layer_{l_idx}.png")
    plt.close()

if fp_med and fp_law and fp_fusion:
    layers_labels, reuse_ratios, new_node_ratios = [], [], []

    for l_idx in TARGET_LAYERS:
        m, l, f = fp_med[l_idx], fp_law[l_idx], fp_fusion[l_idx]
        plot_venn_diagram(l_idx, m, l, f)
        
        union_base = m.union(l)
        intersection = f.intersection(union_base)
        new_nodes = f - union_base
        
        layers_labels.append(f"L{l_idx}")
        reuse_ratios.append(len(intersection) / REP_COUNT * 100)
        new_node_ratios.append(len(new_nodes) / REP_COUNT * 100)

    # 종합 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(layers_labels, reuse_ratios, marker='o', label='Reuse Ratio (Med ∪ Law)', color='royalblue', linewidth=2)
    plt.plot(layers_labels, new_node_ratios, marker='s', label='Fusion-Unique Node Ratio', color='crimson', linewidth=2)
    plt.title(f'Domain Knowledge Transition (Qwen2.5-14B)', fontsize=15)
    plt.xlabel('Model Layers', fontsize=12)
    plt.ylabel('Top-K Node Percentage (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig('qwen14b_transition_analysis.png')
    
    print("\n[Analysis Complete]")
    print(f"- Transition Graph: qwen14b_transition_analysis.png")
    print(f"- Venn Diagrams saved for layers: {TARGET_LAYERS}")