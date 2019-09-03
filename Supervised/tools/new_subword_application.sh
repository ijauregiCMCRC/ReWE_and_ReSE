
TRAIN_VAL_PATH=IWSLT_2016/english_french/en-fr
TEST_PATH=IWSLT_2016/en-fr_test_2015_2016
L1=en
L2=fr
BPE_PATH=IWSLT_2016/english_french/en-fr_BPE


subword-nmt learn-joint-bpe-and-vocab --input $TRAIN_VAL_PATH/train.tok.$L1 $TRAIN_VAL_PATH/train.tok.$L2 -s 32000 -o $BPE_PATH/BPE_32000_model --write-vocabulary $BPE_PATH/vocab.$L1 $BPE_PATH/vocab.$L2

#Apply to train
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L1 --vocabulary-threshold 50 < $TRAIN_VAL_PATH/train.tok.$L1 > $BPE_PATH/train.tok.BPE_32000.$L1
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L2 --vocabulary-threshold 50 < $TRAIN_VAL_PATH/train.tok.$L2 > $BPE_PATH/train.tok.BPE_32000.$L2

#Apply to val
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L1 --vocabulary-threshold 50 < $TRAIN_VAL_PATH/test_2013_plus_2014.tok.$L1 > $BPE_PATH/test_2013_plus_2014.tok.BPE_32000.$L1
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L2 --vocabulary-threshold 50 < $TRAIN_VAL_PATH/test_2013_plus_2014.tok.$L2 > $BPE_PATH/test_2013_plus_2014.tok.BPE_32000.$L2

#Apply to test
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L1 --vocabulary-threshold 50 < $TEST_PATH/test_2015_plus_2016.tok.$L1 > $BPE_PATH/test_2015_plus_2016.tok.BPE_32000.$L1
subword-nmt apply-bpe -c $BPE_PATH/BPE_32000_model --vocabulary $BPE_PATH/vocab.$L2 --vocabulary-threshold 50 < $TEST_PATH/test_2015_plus_2016.tok.$L2 > $BPE_PATH/test_2015_plus_2016.tok.BPE_32000.$L2
