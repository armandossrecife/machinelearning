# Aprendizagem Automática

Projeto de Aprendizagem de Máquina

Fazer a análise do dataset de projetos do Kickstarter.

![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg) 

Dataset: dados/ks-projects-201801.csv

Obs: Dataset baseado no conjunto de dados disponíveis em https://www.kaggle.com/kemical/kickstarter-projects

Caso você não tenha um pacote instalado na sua instância do Python3 use o pip3 para instalar o pacote.

Exemplo: pip3 install nome_do_pacote

## KMEANS
 Faz o agrupamento dos projetos.

### Executar o programa
```
python3 k-means/main.py dados/ks-projects-201801.csv
```
## Multilayer Perceptron
  Faz a classificação dos projetos de acordo com o status de sucesso ou não sucesso.

### Executar o programa
```
jupyter nbconvert --to script kickstarter.ipynb
python3 mlp/kickstarter.py
```
## Gera gráficos referentes ao dataset
   Gera gráficos refentes as análises da base de dados do Kickstarter.

### Executar o programa
```
python3 graficos/graficos-kickstarter.py
```
## Preparação de ambiente:

Comando para criar ambientes usando python3:
```
virtualenv -p python3 ./env
```
Comando para ativar o ambiente virtual:
```
source ./env/bin/activate
```
Comando para criar o arquivo de dependências do projeto:
```
pip freeze > requirements.txt
```
Comando para instalar dependências do projeto:
```
pip install -r requirements.txt
```
### Dependências
 - python3
 - scikit-learn
 - pandas
 - numpy
 - matplotlib
 - virtualenv
 - jupyter
