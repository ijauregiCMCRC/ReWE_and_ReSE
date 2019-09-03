import os

refs_file=open('IWSLT_2016/en-fr_test_2015_2016/test_2015_plus_2016.tok.fr')
hyps_BASELINE_file=open('IWSLT_2016/english_french/models/BASELINE_PRE_EMBS_neubig_style_training/SEED_models/seed_8/pred_5_test.txt')
hyps_NLL_COS_file=open('IWSLT_2016/english_french/models/NLL_COS_PRE_EMBS_DEC_FIX_Lin_200_RELU_Lin_neubig_style_training/SEED_models/lambda_20/seed_8/pred_5_test.txt')


list_refs=[]
for sen in refs_file:
    list_refs.append(sen)
list_hyps_BASELINE=[]
for sen in hyps_BASELINE_file:
    list_hyps_BASELINE.append(sen)
list_hyps_NLL_COS=[]
for sen in hyps_NLL_COS_file:
    list_hyps_NLL_COS.append(sen)

print (len(list_refs))
print (len(list_hyps_BASELINE))
print (len(list_hyps_NLL_COS))

baseline_BLEUS=[]
nll_cos_BLEUS=[]
for i in range(len(list_refs)):
    #Create tmp_ref.txt, tmp_hyp_BASELINE.txt and tmp_hyp_NLL_COS.txt files
    write_ref=open('SINGLE_BLEU/tmp_ref.txt','w')
    write_ref.write(list_refs[i])
    write_ref.close()
    write_hyp_BASELINE = open('SINGLE_BLEU/tmp_hyp_BASELINE.txt', 'w')
    write_hyp_BASELINE.write(list_hyps_BASELINE[i])
    write_hyp_BASELINE.close()
    write_hyp_NLL_COS = open('SINGLE_BLEU/tmp_hyp_NLL_COS.txt', 'w')
    write_hyp_NLL_COS.write(list_hyps_NLL_COS[i])
    write_hyp_NLL_COS.close()
    #COMPUTE BLUE SCORES
    #BASELINE
    os.system('perl tools/multi-bleu.perl SINGLE_BLEU/tmp_ref.txt < SINGLE_BLEU/tmp_hyp_BASELINE.txt > SINGLE_BLEU/tmp_result.txt')
    bleu_file=open('SINGLE_BLEU/tmp_result.txt')
    bleu_line=bleu_file.readline()
    bleu_score=float(bleu_line.split()[2].replace(',',''))
    baseline_BLEUS.append(bleu_score)
    #NLL_COS
    os.system('perl tools/multi-bleu.perl SINGLE_BLEU/tmp_ref.txt < SINGLE_BLEU/tmp_hyp_NLL_COS.txt > SINGLE_BLEU/tmp_result.txt')
    bleu_file = open('SINGLE_BLEU/tmp_result.txt')
    bleu_line = bleu_file.readline()
    bleu_score = float(bleu_line.split()[2].replace(',', ''))
    nll_cos_BLEUS.append(bleu_score)

print (baseline_BLEUS)
print ("\n\n\n")
print (nll_cos_BLEUS)
print (len(baseline_BLEUS))
print (len(nll_cos_BLEUS))

print (sum( x>0 for x in baseline_BLEUS))
print (sum(x>0 for x in nll_cos_BLEUS))

file_summary=open('SINGLE_BLEU/results_sumary.txt','w')
for j in range(len(baseline_BLEUS)):
    bas_score=baseline_BLEUS[j]
    nll_cos_score=nll_cos_BLEUS[j]
    if bas_score==0 and nll_cos_score==0:
        file_summary.write("Line: "+str(j+1)+"  Baseline_score: "+str(bas_score)+"  NLL_COS_score: "+str(nll_cos_score)+"  ZERO_DRAW\n")
    elif bas_score > nll_cos_score:
        file_summary.write("Line: " + str(j + 1) + "  Baseline_score: " + str(bas_score) + "  NLL_COS_score: " + str(nll_cos_score) + "  BASELINE_WINS\n")
    elif bas_score < nll_cos_score:
        file_summary.write("Line: " + str(j + 1) + "  Baseline_score: " + str(bas_score) + "  NLL_COS_score: " + str(nll_cos_score) + "  NLL_COS_WINS\n")
    else:
        file_summary.write("Line: " + str(j + 1) + "  Baseline_score: " + str(bas_score) + "  NLL_COS_score: " + str(nll_cos_score) + "  NONZERO_DRAW\n")