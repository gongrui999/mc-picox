"""Analyze recommend.xlsx for PICO entity alignment feasibility."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import openpyxl

wb = openpyxl.load_workbook(r'D:\PICOX\recommend.xlsx', read_only=True)
ws = wb['Sheet1']

total = 0
has_p = 0; has_i = 0; has_o = 0
p_texts = []; i_texts = []; o_texts = []
final_lengths = []

for row in ws.iter_rows(min_row=2, values_only=True):
    if row[0] is None:
        continue
    total += 1
    final = str(row[22] or '')
    final_lengths.append(len(final))

    disease = str(row[1] or ''); age = str(row[3] or ''); gender = str(row[4] or '')
    comorbidity = str(row[6] or '')
    has_p_flag = any(v and v != 'null' for v in [disease, age, gender, comorbidity])
    if has_p_flag:
        has_p += 1

    interv_name = str(row[10] or ''); interv_detail = str(row[12] or '')
    has_i_flag = any(v and v != 'null' for v in [interv_name, interv_detail])
    if has_i_flag:
        has_i += 1

    outcome_ind = str(row[14] or ''); outcome_tgt = str(row[15] or '')
    has_o_flag = any(v and v != 'null' for v in [outcome_ind, outcome_tgt])
    if has_o_flag:
        has_o += 1

    if total <= 200:
        if disease and disease != 'null' and final:
            p_texts.append((disease, disease in final, final[:100]))
        if interv_name and interv_name != 'null' and final:
            i_texts.append((interv_name, interv_name in final, final[:100]))
        if outcome_ind and outcome_ind != 'null' and final:
            o_texts.append((outcome_ind, outcome_ind in final, final[:100]))

print(f'Total rows: {total}')
print(f'Has P info: {has_p} ({has_p/total*100:.1f}%)')
print(f'Has I info: {has_i} ({has_i/total*100:.1f}%)')
print(f'Has O info: {has_o} ({has_o/total*100:.1f}%)')
print(f'Final text avg length: {sum(final_lengths)/len(final_lengths):.0f} chars')
print(f'Final text max length: {max(final_lengths)} chars')
print()

p_match = sum(1 for _, m, _ in p_texts if m)
i_match = sum(1 for _, m, _ in i_texts if m)
o_match = sum(1 for _, m, _ in o_texts if m)
print(f'=== Exact string match rate (first 200 rows) ===')
print(f'P (Disease in Final):        {p_match}/{len(p_texts)} = {p_match/max(len(p_texts),1)*100:.1f}%')
print(f'I (Interv_Name in Final):    {i_match}/{len(i_texts)} = {i_match/max(len(i_texts),1)*100:.1f}%')
print(f'O (Outcome_Ind in Final):    {o_match}/{len(o_texts)} = {o_match/max(len(o_texts),1)*100:.1f}%')
print()

print('=== P alignment samples ===')
for text, match, final in p_texts[:5]:
    print(f'  Disease="{text}"  match={match}')
    print(f'    Final: {final}...')
print()
print('=== I alignment samples ===')
for text, match, final in i_texts[:5]:
    print(f'  Interv="{text}"  match={match}')
    print(f'    Final: {final}...')
print()
print('=== O alignment samples ===')
for text, match, final in o_texts[:5]:
    print(f'  Outcome="{text}"  match={match}')
    print(f'    Final: {final}...')

print()
print('=== Failed alignment examples ===')
count = 0
for text, match, final in i_texts + o_texts + p_texts:
    if not match and count < 10:
        print(f'  Field="{text}"')
        print(f'    Final: {final}...')
        count += 1
