export PYTHONPATH=.
export TK_SILENCE_DEPRECATION=1

for i in 1 2 3 4 5 6 7 8 9 10
do
    python experiments/learn_eq.py run$i
    python experiments/learn_smk.py run$i
done
