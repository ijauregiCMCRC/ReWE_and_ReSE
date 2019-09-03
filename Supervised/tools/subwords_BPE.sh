# Generate Subword Units (BPE)
# Clone Subword NMT
#if [ ! -d "subword_BPE/subword-nmt" ]; then
#  git clone https://github.com/rsennrich/subword-nmt.git "subword_BPE/subword-nmt"
#fi

# Learn Shared BPE
#for merge_ops in 8000; do
#  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
#  cat "${OUTPUT_DIR}/train.tok.clean.en" "${OUTPUT_DIR}/train.tok.clean.eu" | \
#    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"
#  cat "data/eu-en_IT/train.en.atok" "data/eu-en_IT/train.eu.atok" | \
#    tools/subword_BPE/subword-nmt/learn_bpe.py -s $merge_ops > "data/eu-en_IT/BPE_EN_EU.${merge_ops}"

#  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
#  for lang in en eu; do
#  for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
#  outfile="${f%.*}.bpe.${merge_ops}.en"

#TRAIN
tools/subword_BPE/subword-nmt/apply_bpe.py -c "../OpenNMT_Last_new/IWSLT_2016/czech-english/cs-en/BPE_CS_EN_AGAIN.32000" < "exp_ofd/cs_en/out_of_domain/IT_test.tok.cs" > "exp_ofd/cs_en/out_of_domain/IT_test.tok.bpe_32000.cs"
tools/subword_BPE/subword-nmt/apply_bpe.py -c "../OpenNMT_Last_new/IWSLT_2016/czech-english/cs-en/BPE_CS_EN_AGAIN.32000" < "exp_ofd/cs_en/out_of_domain/IT_test.tok.en" > "exp_ofd/cs_en/out_of_domain/IT_test.tok.bpe_32000.en"
#DEV
#tools/subword_BPE/subword-nmt/apply_bpe.py -c "data/eu-en_IT/BPE_EN_EU.${merge_ops}" < "data/eu-en_IT/val.eu.atok" > "data/eu-en_IT/val.eu.atok.bpe_${merge_ops}"
#tools/subword_BPE/subword-nmt/apply_bpe.py -c "data/eu-en_IT/BPE_EN_EU.${merge_ops}" < "data/eu-en_IT/val.en.atok" > "data/eu-en_IT/val.en.atok.bpe_${merge_ops}"
#TEST
#tools/subword_BPE/subword-nmt/apply_bpe.py -c "data/eu-en_IT/BPE_EN_EU.${merge_ops}" < "data/eu-en_IT/test.eu.atok" > "data/eu-en_IT/test.eu.atok.bpe_${merge_ops}"
#tools/subword_BPE/subword-nmt/apply_bpe.py -c "data/eu-en_IT/BPE_EN_EU.${merge_ops}" < "data/eu-en_IT/test.en.atok" > "data/eu-en_IT/test.en.atok.bpe_${merge_ops}"



#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/dev.src" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/dev_plus_500K_bpe.src"
#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/test.src" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/test_plus_500K_bpe.src"

#MT
#subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/official_23K/en_de_bpe.${merge_ops}" < "WMT17_APE/BPE_data/official_23K/train.mt" > "WMT17_APE/BPE_data/official_23K/train_bpe.mt"
#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/dev.mt" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/dev_plus_500K_bpe.mt"
#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/test.mt" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/test_plus_500K_bpe.mt"

#PE
##tools/subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_news_data/Parallel_data/Europarl_tok_50l_subword_model.${merge_ops}" < "WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.50l.de" > "WMT17_news_data/Parallel_data/europarl-v7.de-en.tok.50l.bpe.de"
#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/dev.pe" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/dev_plus_500K_bpe.pe"
#  subword_BPE/subword-nmt/apply_bpe.py -c "WMT17_APE/BPE_data/cross_lingual_official/en_de_bpe.${merge_ops}" < "WMT17_APE/SHARED_TASK_EXTRA_500K/files/test.pe" > "WMT17_APE/BPE_data/cross_lingual_official_big_training/test_plus_500K_bpe.pe"
#  echo ${outfile}
#  done
#  done

  # Create vocabulary file for BPE

#SRC
#  cat "WMT17_APE/BPE_data/cross_lingual_official/train_bpe.src" "WMT17_APE/BPE_data/cross_lingual_official/dev_bpe.src" | \
#    subword_BPE/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "WMT17_APE/BPE_data/cross_lingual_official/vocab_src.bpe.${merge_ops}"

#MT
#  cat "WMT17_APE/BPE_data/cross_lingual_official/train_bpe.mt" "WMT17_APE/BPE_data/cross_lingual_official/dev_bpe.mt" | \
#    subword_BPE/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "WMT17_APE/BPE_data/cross_lingual_official/vocab_mt.bpe.${merge_ops}"

#PE
#  cat "WMT17_APE/BPE_data/cross_lingual_official/train_bpe.pe" "WMT17_APE/BPE_data/cross_lingual_official/dev_bpe.pe" | \
#    subword_BPE/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "WMT17_APE/BPE_data/cross_lingual_official/vocab_pe.bpe.${merge_ops}"

#done

echo "All done."
