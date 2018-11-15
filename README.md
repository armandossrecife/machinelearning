# Aprendizagem Automática

Esboço inicial do Projeto da Disciplina de Aprendizagem de Máquina

Fazer a análise do dataset de projetos do Kickstarter.

Dataset: ks-projects-201801.csv

Obs: Dataset baseado no conjunto de dados disponíveis em https://www.kaggle.com/kemical/kickstarter-projects

Caso você não tenha um pacote instalado na sua instância do Python3 use o pip3 para instalar o pacote.

Exemplo: pip3 install nome_do_pacote

## Preparação de ambiente:

Pré-requisitos:
Python 3
Pacote pip
Pacote virtualenv

Comando para criar ambientes usando python3:
virtualenv -p python3 ./env

Comando para ativar o ambiente virtual:
source ./env/bin/activate

Comando para criar o arquivo de dependências do projeto:
pip freeze > requirements.txt

Comando para instalar dependências do projeto:
pip install -r requirements.txt
