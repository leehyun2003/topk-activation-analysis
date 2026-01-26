import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
import re
import os

# 1. 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-14B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto"
)

TARGET_LAYERS = [24]
domains = ['medical', 'legal', 'fusion']
data_files = {
    'medical': 'data/medical.txt',
    'legal': 'data/law.txt',
    'fusion': 'data/medical_law.txt'
}

# 저장소: 최대 활성값, 그때의 문장, 그리고 핵심 '토큰' 저장
node_max_val = {d: defaultdict(float) for d in domains}
node_max_data = {d: defaultdict(lambda: {"sent": "", "token": ""}) for d in domains}

# 2. Hook 정의 (토큰 위치 추적 로직 추가)
def get_max_val_hook(l_idx, current_domain):
    def hook(module, input, output):
        # input[0] shape: [1, seq_len, intermediate_dim]
        h = input[0].detach() 
        
        # 각 노드(차원)별로 문장 내에서 가장 높은 활성값(vals)과 그 위치(indices)를 찾음
        vals, indices = torch.max(h[0], dim=0) 
        
        # 현재 입력된 문장의 실제 토큰 리스트 변환
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        for n_idx in range(h.size(-1)):
            v = vals[n_idx].item()
            if v > node_max_val[current_domain][n_idx]:
                node_max_val[current_domain][n_idx] = v
                # 최댓값이 발생한 위치의 토큰 추출
                t_idx = indices[n_idx].item()
                node_max_data[current_domain][n_idx] = {
                    "sent": current_sentence,
                    "token": all_tokens[t_idx].replace(' ', '') # Qwen 토큰 특수기호 제거
                }
    return hook

# 3. 데이터 분석 실행
for d_name, path in data_files.items():
    if not os.path.exists(path):
        continue
    
    print(f">>> {d_name} 분석 중...")
    handle = model.model.layers[TARGET_LAYERS[0]].mlp.down_proj.register_forward_hook(get_max_val_hook(TARGET_LAYERS[0], d_name))
    
    with open(path, 'r', encoding='utf-8') as f:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', f.read().replace('\n', ' ')) if len(s.strip()) > 5][:50]

    for sent in sentences:
        current_sentence = sent
        # Hook에서 사용하기 위해 input_ids를 전역에 가깝게 유지
        inputs = tokenizer(sent, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        with torch.no_grad():
            model(**inputs)
    handle.remove()

# 4. 결과 출력
print("\n" + "="*80)
print(" [지식 영토 증명] 토큰 단위 정밀 분석 (Layer 24)")
print("="*80)

PURITY_LIMIT = 0.6
MIN_STRENGTH = 2.0

for d_target in domains:
    print(f"\n● [{d_target.upper()}] 영역 전문 노드 및 반응 토큰")
    found_count = 0
    
    # 해당 도메인 고점 노드 정렬
    sorted_nodes = sorted(node_max_val[d_target].keys(), key=lambda x: node_max_val[d_target][x], reverse=True)

    for n_idx in sorted_nodes:
        v_m = node_max_val['medical'][n_idx]
        v_l = node_max_val['legal'][n_idx]
        v_f = node_max_val['fusion'][n_idx]
        total = v_m + v_l + v_f + 1e-9
        
        # 타겟 도메인 순도 계산
        p = (v_m if d_target == 'medical' else v_l if d_target == 'legal' else v_f) / total
        
        if p > PURITY_LIMIT and node_max_val[d_target][n_idx] > MIN_STRENGTH:
            res = node_max_data[d_target][n_idx]
            print(f"노드 #{n_idx:<5} | 순도: {p:.2f} | 강도: {node_max_val[d_target][n_idx]:.2f}")
            # 수정 전: print(f"핵심 토큰: ➔ '{res['token']}'")
            # 수정 후 (바이트 복원)
            token_bytes = tokenizer.convert_tokens_to_ids(res['token'])
            clean_token = tokenizer.decode([token_bytes])
            print(f"핵심 토큰: ➔ '{clean_token}'")
            print(f"   문장 맥락: \"{res['sent']}\"")
            found_count += 1
            if found_count >= 5: break