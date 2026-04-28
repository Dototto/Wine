# Wine

# Analisi e Classificazione del Dataset Wine

Questo progetto illustra il workflow completo di Machine Learning applicato al celebre **Wine Dataset**, disponibile tramite la libreria **Scikit-Learn**.

Il processo include l'analisi esplorativa, il preprocessing dei dati, la **cross validation** e l'addestramento di due modelli di classificazione: **Regressione Logistica** e **k-Nearest Neighbors (k-NN)**.

---

## Dataset

È stato utilizzato il famoso dataset Wine, che contiene analisi chimiche di vini prodotti da tre diverse cultivar.

Le feature presenti nel dataset sono:

* alcohol  
* malic_acid  
* ash  
* alcalinity_of_ash  
* magnesium  
* total_phenols  
* flavanoids  
* nonflavanoid_phenols  
* proanthocyanins  
* color_intensity  
* hue  
* od280/od315_of_diluted_wines  
* proline  

Obiettivo: classificare il vino in una delle tre classi:

* Class 0  
* Class 1  
* Class 2

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

---

## Modelli di Classificazione: Logistic Regression & k-NN

Questo repository contiene una panoramica teorica e pratica su due dei principali algoritmi di Machine Learning utilizzati per la classificazione: la **Regressione Logistica Multinomiale** e i **k-Nearest Neighbors (k-NN)**.

---

## 1. Regressione Logistica Multinomiale (Softmax Regression)

La Regressione Logistica Multinomiale è un'estensione della regressione logistica binaria, progettata per prevedere la probabilità di appartenenza di un input a una tra **tre o più classi** distinte.

### Teoria
A differenza del caso binario (dove si usa la funzione Sigmoide), questo modello utilizza la funzione **Softmax**. Per ogni classe $j$, il modello calcola un punteggio lineare; la funzione Softmax trasforma poi questi punteggi in probabilità:

### Teoria

$$
P(y=j|\mathbf{x}) = \frac{e^{\mathbf{w}_j^T \mathbf{x}}}{\sum_{k=1}^{K} e^{\mathbf{w}_k^T \mathbf{x}}}
$$

Dove:

* **$\mathbf{x}$** è il vettore delle feature chimiche  
* **$\mathbf{w}_j$** sono i pesi della classe $j$  
* **$K=3$** è il numero di classi  

Per comprendere come il modello giunga alla classificazione, analizziamo i simboli utilizzati:

*   **$P(y=j | \mathbf{x})$**: Rappresenta la **probabilità condizionata**. È la probabilità che il campione appartenga alla classe $j$ (es. "Class 0"), date le sue caratteristiche misurate $\mathbf{x}$.
*   **$\mathbf{x}$**: È il **vettore delle caratteristiche** (input). Nel caso del dataset Wine, contiene le caratteristiche chimiche del vino, come alcohol, malic_acid, ash, magnesium, flavanoids, color_intensity e proline.
*   **$\mathbf{w}_j$**: Rappresenta il **vettore dei pesi** (o coefficienti) che il modello ha imparato specificamente per la classe $j$. Ogni classe ha il suo set di pesi.
*   **$\mathbf{w}_j^T \mathbf{x}$**: È il **prodotto scalare** tra pesi e caratteristiche. Rappresenta il "punteggio grezzo" (chiamato *Logit*) assegnato alla classe $j$. Più alto è questo valore, più il modello "crede" che il campione appartenga a quella classe.
*   **$e$**: È la costante di Nepero (circa 2.718). Elevare $e$ al punteggio garantisce che il risultato sia sempre un **numero positivo** e accentua lo stacco tra la classe dominante e le altre.
*   **$\sum_{k=1}^{K}$**: Rappresenta la **sommatoria** calcolata su tutte le classi possibili ($K=3$ nel dataset Wine). Dividere per questa somma serve a **normalizzare** i risultati, facendo sì che la somma di tutte le probabilità finali sia esattamente 1 (o 100%).

Il risultato è un vettore di probabilità la cui somma è sempre pari a 1 (100%).

*   **Punti di forza:** Fornisce probabilità di classificazione, è veloce da addestrare e facile da interpretare (tramite i coefficienti). 

---

## 2. k-Nearest Neighbors (k-NN)

Il k-NN è un algoritmo di tipo **non parametrico** e **lazy learning** (apprendimento pigro). Non costruisce un modello matematico esplicito durante l'addestramento, ma memorizza semplicemente i dati di training.

### Teoria
Il principio cardine è la **similitudine geometrica**. Quando riceve un nuovo punto da classificare:
1. Calcola la **distanza** (solitamente Euclidea) tra il nuovo punto e tutti quelli presenti nel set di addestramento.
2. Individua i **$k$ punti più vicini**.
3. Assegna la classe basandosi sulla **maggioranza** (voto ponderato o semplice) tra i $k$ vicini.

<img width="591" height="515" alt="image" src="https://github.com/user-attachments/assets/215e426e-d671-410d-9995-6c46f5c26b8c" />


*   **Punti di forza:** Estremamente semplice da implementare, si adatta a decision boundary molto complessi e non lineari.
*   **Punti di debolezza:** Molto costoso in termini computazionali durante la fase di test (lento con dataset grandi) e sensibile alla scala delle caratteristiche (richiede normalizzazione).

---

## Confronto Rapido


| Caratteristica | Logistica Multinomiale | k-Nearest Neighbors (k-NN) |
| :--- | :--- | :--- |
| **Tipo di Modello** | Parametrico (Lineare) | Non Parametrico (Basato su istanze) |
| **Fase di Training** | Più lenta (ottimizzazione) | Istantanea (memorizzazione) |
| **Fase di Predizione** | Molto veloce | Lenta (calcolo distanze) |
| **Outlier** | Robusta | Molto sensibile |

---

# Descrizione del Processo

## 1. Analisi Esplorativa (EDA)

### 1. Analisi Esplorativa (EDA)
Prima dell'addestramento, sono state effettuate le seguenti analisi:
*   **Gestione Valori Mancanti:** Verifica della presenza di dati nulli all'interno delle feature per garantire l'integrità del dataset.
*   **Analisi di Correlazione:** Generazione di una **Heatmap** per visualizzare la correlazione di Pearson tra le diverse caratteristiche (lunghezza/larghezza di sepali e petali).
*   **Rimozioni feature altamente correlate:** Droppare una delle due feature con un valore assoluto di correlazione superiore a una certa soglia per evitare la multicollinearità.
<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/7dabc2b2-b214-4a9f-b3c8-1040cd2fa5da" />

---

### 2. Preprocessing dei Dati
Il dataset è stato preparato seguendo questi passaggi:
*   **Separazione:** Divisione dei dati in **Features** ($X$) e **Target** ($y$).
*   **Splitting:** Suddivisione del dataset in set di Training e Test con una proporzione **80/20** (80% per l'apprendimento, 20% per la validazione).
*   **Standardizzazione:** Applicazione di `StandardScaler` per scalare le feature in modo che abbiano media 0 e varianza 1. Questo passaggio è fondamentale soprattutto per il modello k-NN, basato sul calcolo delle distanze.

### 3. Modelli di Machine Learning

Sono stati implementati e confrontati due classificatori:

#### A. Regressione Logistica (Multinomiale)
*   **Configurazione:** È stato utilizzato il solver `lbfgs` (*Limited-memory Broyden–Fletcher–Goldfarb–Shanno*).
*   **Motivazione:** È l'algoritmo di ottimizzazione predefinito in Scikit-Learn per problemi di piccole e medie dimensioni. È stato scelto perché supporta nativamente la perdita multinomiale (Cross-Entropy) ed è molto efficiente nel gestire dataset con poche osservazioni, garantendo una convergenza rapida rispetto ad altri solver come `liblinear`.
*   **Logica:** Il modello gestisce intrinsecamente la classificazione multi-classe calcolando le probabilità per le tre specie di Iris.

#### B. k-Nearest Neighbors (k-NN)
*   **Parametri:** Impostato con **$k = 11$**.
*   **Motivazione della scelta:** 
    1.  **Regola della Radice Quadrata:** Una pratica comune (euristica) consiste nello scegliere $k \approx \sqrt{n}$, dove $n$ è il numero di campioni nel training set. Con 142 campioni (80% di 178), $\sqrt{142} \approx 11.91$, da cui il valore 11.
    2.  **Valore Dispari:** Si è scelto un numero dispari per evitare situazioni di "pareggio" (tie) durante la votazione delle classi, garantendo che ci sia sempre una classe di maggioranza netta.
*   **Logica:** La classificazione avviene in base alla classe di maggioranza degli 11 campioni più vicini nel set di training.

---

## 4. Cross Validation dei Modelli

Per valutare la robustezza dei modelli è stata applicata la **K-Fold Cross Validation** con **cv = 5**.

### Cos'è la Cross Validation

Il dataset viene suddiviso in 5 blocchi:

* ad ogni iterazione, 4 blocchi vengono usati per il training  
* 1 blocco viene usato per il test  
* il processo viene ripetuto 5 volte cambiando il blocco di validazione  

In questo modo ogni osservazione viene usata sia per training che per validazione.

### Vantaggi

* stima più affidabile delle performance  
* minore dipendenza da un singolo train/test split  
* migliore controllo dell'overfitting  
* confronto più stabile tra modelli


## Risultati
I modelli vengono valutati in base a:
*   **Accuracy:** La percentuale totale di previsioni corrette.
*   **Precision:** La capacità del modello di non classificare come positiva una risposta che in realtà è negativa (evitare i "falsi positivi").
*   **Recall (Sensibilità):** La capacità del modello di individuare tutti i casi positivi reali (evitare i "falsi negativi").
*   **F1-Score:** La media armonica tra Precision e Recall, utile per avere un unico indicatore che bilanci entrambi gli aspetti, specialmente se le classi fossero sbilanciate.

## Requisiti
Per implementare questi modelli in Python, si consiglia l'uso di:
- `scikit-learn`
- `numpy`
- `pandas`
- `Seaborn & Matplotlib`

---

## API con FastAPI

Il modello è stato integrato in una API REST utilizzando FastAPI.

### Endpoint principale:

```bash
POST /predict
```

### Esempio richiesta:
```json
{
  "alcohol": 13.5,
  "malic_acid": 2.1,
  "ash": 2.4,
  "alcalinity_of_ash": 18.0,
  "magnesium": 105,
  "flavanoids": 2.9,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.8,
  "color_intensity": 5.2,
  "hue": 1.0,
  "od280_od315_of_diluted_wines": 3.1,
  "proline": 950
}
```

### Esempio risposta:
```json
{
  "prediction_logistic_regression": "class_0",
  "prediction_knn": "class_0"
}
```

---

## Docker

L'applicazione è stata containerizzata utilizzando Docker.

### Build immagine:

```bash
docker build -t wine-image .
```

### Avvio container:

```bash
docker run -p 8002:8000 wine-image
```

---

## Workflow Git

Il progetto è stato sviluppato collaborativamente utilizzando Git e GitHub:

* creazione repository
* utilizzo di commit, push e pull
* aggiornamenti continui del codice

---

## Tecnologie utilizzate

* Python
* FastAPI
* Scikit-learn
* Docker

---

## Come eseguire il progetto

1. Clonare la repository:

```bash
git clone <repo-url>
cd <repo-name>
```

2. Costruire l'immagine Docker:

```bash
docker build -t wine-image .
```

3. Avviare il container:

```bash
docker run -p 8002:8000 wine-image
```

4. Aprire nel browser:

```
http://localhost:8002/docs
```

---

## Autori

Progetto sviluppato per scopi didattici.
