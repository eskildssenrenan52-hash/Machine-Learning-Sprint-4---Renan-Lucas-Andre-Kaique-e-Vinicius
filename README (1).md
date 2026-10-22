# Challenge Sprint 4 – Classificação linear de Energy_Class

Dataset: Renewable Energy Production Dataset (2010–2020), Kaggle – `Renewable_Energy_Data.csv` (1000 linhas, 11 colunas, sem nulos).
Código: `challenge_sprint4.py` (`pip install pandas numpy scikit-learn matplotlib seaborn` e `python challenge_sprint4.py Renewable_Energy_Data.csv`).
Saídas em `resultados/`: `matriz_correlacao.png/.csv`, `matriz_confusao_Cenario1_60-40.png`, `matriz_confusao_Cenario2_85-15.png`, `metricas.csv`, `resumo.txt`.

## 01) Análise do dataset
**a)** Carregado com `pandas.read_csv`. Classes: High 486, Medium 336, Low 178 (desbalanceado).
**b)** Encoding ordinal de `Energy_Class`: Low=0, Medium=1, High=2. Foi escolhida a ordem natural das classes (e não a alfabética do `LabelEncoder`) para que o sinal da correlação tenha significado. As variáveis categóricas `Region`, `Energy_Source` e `Season` foram transformadas em dummies (`get_dummies`, `drop_first=True`), resultando em 15 features.
**c)** Matriz em `resultados/matriz_correlacao.png`.
**d)** Correlação absoluta com `Energy_Class`: `Efficiency_Ratio` = 0,211; todas as demais ≤ 0,065 (ex.: Region_North 0,065; Rainfall_mm 0,046; Wind_Speed_m_s 0,027; Solar_Radiation 0,008). Entre as features, a maior correlação é Energy_Source_Solar × Energy_Source_Wind (0,49), sem colinearidade severa (> 0,9).
**Decisão: usar um subconjunto, apenas `Efficiency_Ratio`** (critério |r| ≥ 0,10 com o alvo). Justificativa: as outras 14 features têm correlação linear praticamente nula com a classe, ou seja, adicionam ruído e dimensionalidade sem sinal linear que um modelo linear possa explorar. A checagem por validação cruzada (5-fold estratificada) confirma: acurácia média de 0,765 com `Efficiency_Ratio` sozinha contra 0,769 com as 15 features, uma diferença desprezível. O modelo mais simples tem desempenho equivalente, é mais fácil de interpretar e menos propenso a overfitting. Ressalva: correlação de Pearson mede só relação linear com um alvo ordinal; features não lineares ou interações podem existir e não são captadas por um modelo linear.

## 02) Modelo
**a)** `sklearn.linear_model.LogisticRegression` (multiclasse), em pipeline com `StandardScaler`.
**b)** O modelo calcula uma combinação linear das features, z = w·x + b, para cada classe, e converte em probabilidades com softmax. A decisão escolhe a classe de maior score, e a fronteira entre duas classes é o conjunto de pontos onde os scores se igualam, (w₁−w₂)·x + (b₁−b₂) = 0, que é um hiperplano (um ponto com uma única feature, uma reta com duas, um plano com três). Como a fronteira é linear nas features, o modelo é linear; a sigmoide/softmax só transforma o score em probabilidade e não curva a fronteira.
**c/d)** Treino com `Efficiency_Ratio`, `train_test_split(..., random_state=42, stratify=y)` e demais condições idênticas. Cenário 1: 600 treino / 400 teste. Cenário 2: 850 treino / 150 teste.

## 03) Avaliação (precisão e recall = média macro entre as 3 classes)
| Cenário | Acurácia | Precisão | Recall |
|---|---|---|---|
| 1 – 60/40 | 0,7575 | 0,8089 | 0,7209 |
| 2 – 85/15 | 0,6667 | 0,6735 | 0,6471 |

Matrizes de confusão (linhas = real, colunas = previsto; ordem Low, Medium, High):
- Cenário 1: [[44, 24, 3], [0, 92, 42], [0, 28, 167]]
- Cenário 2: [[18, 7, 2], [5, 24, 21], [0, 15, 58]]

**c)** O desempenho é moderado (≈67–76% de acurácia), melhor que o chute da classe majoritária (48,6%), mas insuficiente para uso crítico. Os erros se concentram na classe `Medium`, confundida com `High` (42 de 134 no cenário 1; 21 de 50 no cenário 2) e, em menor grau, com `Low`. No cenário 1, `High` tem recall de 85,6% e `Medium` de 68,7%; no cenário 2, `Medium` cai para 48,0%. Os erros vêm de usar uma única feature e um modelo linear: as classes são ordenadas pela escala de `Efficiency_Ratio`, que tem forte assimetria e outliers (máx. 16.471 contra mediana de 150), e uma fronteira linear sobre a escala bruta separa mal as faixas Medium e High.
**d)** Com 40% de teste a acurácia foi 75,8% e com 15% foi 66,7%, uma queda de 9 pontos percentuais. Isso não significa que treinar com mais dados piorou o modelo: o cenário 2 tem apenas 150 exemplos de teste (27 Low, 50 Medium, 73 High), então a métrica oscila bastante com poucos exemplos. A validação cruzada (≈0,765) indica que o desempenho real está perto do cenário 1.
**e)** Teste maior (40%): estimativa de desempenho mais estável e confiável, com mais exemplos por classe na matriz de confusão, mas menos dados para treino (600), o que pode subestimar o desempenho do modelo final. Teste menor (15%): mais dados para treinar (850) e modelo potencialmente melhor, mas avaliação mais ruidosa, em que poucos exemplos mudam muito as métricas e classes minoritárias ficam sub-representadas. Mitigações: split estratificado, `random_state` fixo e validação cruzada.

## Observação
Uma exploração extra mostrou que aplicar `log` em `Efficiency_Ratio` e combiná-la com as demais features eleva a acurácia média em CV para ≈0,91, ou seja, há margem de melhoria com transformação não linear. Essa melhoria não foi aplicada para manter o escopo do enunciado (seleção por correlação + modelo linear).
