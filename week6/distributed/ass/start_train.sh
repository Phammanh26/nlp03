export DEBUG=0
git clone https://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@github.com/Phammanh26/nlp03.git
pip install -r  $pwd/nlp03/week6/distributed/ass/requirements.txt
cd $pwd/nlp03/week6/distributed/ass"
torchrun --nproc_per_node=2 --master_port=1234 train_with_ddp_solution.py