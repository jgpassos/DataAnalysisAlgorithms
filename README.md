# algoritmosMachineLearning
Algoritmos de Machine Learning usando Python


Neste repositório você encontra a aplicação de alguns algoritmos de Machine Learning para a análise e classificação de algumas bases de dados em determinados casos.

REGRESSÃO:

É um algoritmo supervisionado usado para estimar o valor de algo baseado em uma série de outros dados históricos, portanto olhando para o passado você pode prever o futuro.

Usado para prever algum resultado com base em dados históricos.

É traçada uma linha reta para analisar a relação entre os dados de duas ou mais variáveis.

Correlação:
mede a correlação linear entre a nuvem de pontos
-1 - correlação linear perfeita negativa
1 - correlação linear perfeita positiva
0 - não tem correlação linear

Classificação -> diferencia o tipo do objeto em relação a sua classe
Regressão -> o mapeamento é feito por seus valores contínuos

ÁRVORE DE DECISÃO:

Uma árvore de decisão é um algoritmo de aprendizado de máquina supervisionado que é utilizado para classificação e para regressão.

MATRIZ DE CONFUSÃO:

Uma matriz confusão é um resumo tabular do número de previsões corretas e incorretas feitas por um classificador. Ele pode ser usado para avaliar o desempenho de um modelo de classificação através do cálculo de métricas de desempenho como accuracy, precision, recal.

- Variável preditora: é aquela que será passada para o modelo, tendo influência na variável que queremos encontrar. Por exemplo: Se queremos prever as vendas de sorvete, a estação do ano pode interferir nas vendas.

- Variável alvo: é a variável que queremos prever. No exemplo acima seria as vendas de sorvete.

DECISION TREE REGRESSION:

As árvores de decisão são usadas para ajustar uma curva senoidal com adição de observação ruidosa. Como resultado, ele aprende regressões lineares locais aproximando a curva senoidal.

A profundidade máxima da árvore (controlada pelo max_depth parâmetro) se definida muito alta, as árvores de decisão aprendem detalhes muito finos dos dados de treinamento e aprendem com o ruído, ou seja, elas se ajustam excessivamente.

A medida que esse parâmetro é fixado, podemos treiná-lo e testá-lo para verificar quão bons são os resultados.

--------------------------------------------------------

BASES DE DADOS UTILIZADAS:

- Breast Cancer Wisconsin (Prognostic) Data Set

Cada registro representa dados de acompanhamento de um caso de câncer de mama. Estes são pacientes consecutivos atendidos pelo Dr. Wolberg desde 1984, e incluem apenas os casos que apresentam câncer de mama invasivo e nenhuma evidência de metástases à distância no momento do diagnóstico.

Cada registro representa dados de acompanhamento de um caso de câncer de mama.

Número de instâncias: 198
Número de atributos: 34

- Auto MPG Data Set

Os dados dizem respeito ao consumo de combustível do ciclo da cidade em milhas por galão, a ser previsto em termos de 3 atributos discretos multivalorados e 5 contínuos.

Número de instâncias: 398
Atributos:
    1. mpg: contínuo
    2. cilindros: discreto multivalorado
    3. deslocamento: contínuo
    4. potência: contínua
    5. peso: contínuo
    6. aceleração: contínua
    7. ano do modelo: discreto multivalorado
    8. origem: discreto multivalorado
    9. nome do carro: string (exclusivo para cada instância)




