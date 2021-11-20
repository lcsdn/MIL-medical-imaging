# Deep multi-instance learning of cancerous lymphocytosis

Joint work with Célia C.. Code for challenge of the course "Deep Learning for Medical Imaging" taught in the MVA master in 2020/2021. The challenge consisted in using multi-instance learning to diagnosise cancerous lymphocytosis using bags of blood smear images.

Link of the challenge: https://www.kaggle.com/c/3md3070-dlmi

Achieved rank: 2/43

## Usage

1. To reproduce our results, please install the requirements indicated in ``requirements.txt`` and add the data downloaded from Kaggle to the ``data`` folder.

2. Then, simply run ``python reproduce.py`` in the shell, or ``!python reproduce.py`` if you are using Google Colab (which we used).

3. After execution, there will be a saved file ``predictions.csv``, which can then be submitted to Kaggle.
