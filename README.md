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

---

## Modelli di Classificazione: Logistic Regression & k-NN

Questo repository contiene una panoramica teorica e pratica su due dei principali algoritmi di Machine Learning utilizzati per la classificazione: la **Regressione Logistica Multinomiale** e i **k-Nearest Neighbors (k-NN)**.

---

## 1. Regressione Logistica Multinomiale (Softmax Regression)

La Regressione Logistica Multinomiale è un'estensione della regressione logistica binaria, progettata per prevedere la probabilità di appartenenza di un input a una tra **tre o più classi** distinte.

### Teoria

$$
P(y=j|\mathbf{x}) = \frac{e^{\mathbf{w}_j^T \mathbf{x}}}{\sum_{k=1}^{K} e^{\mathbf{w}_k^T \mathbf{x}}}
$$

Dove:

* **$\mathbf{x}$** è il vettore delle feature chimiche  
* **$\mathbf{w}_j$** sono i pesi della classe $j$  
* **$K=3$** è il numero di classi  

### Punti di forza

* Fornisce probabilità di classificazione  
* Rapida in training e prediction  
* Interpretabile  

---

## 2. k-Nearest Neighbors (k-NN)

Il k-NN è un algoritmo **non parametrico** basato sulla vicinanza geometrica tra osservazioni.

### Funzionamento

1. Calcola la distanza dal nuovo punto ai campioni del training set  
2. Seleziona i **k vicini più prossimi**  
3. Restituisce la classe più frequente  

### Punti di forza

* Semplice  
* Adatto a pattern non lineari  

### Limiti

* Più lento in prediction  
* Sensibile alla scala delle feature  

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
*   **Motivazione:** È l'algoritmo di ottimizzazione predefinito in Scikit-Learn per problemi di piccole e medie dimensioni. È stato scelto perché supporta nativamente la perdita multinomiale (Cross-Entropy) ed è molto efficiente nel gestire dataset con poche osservazioni come Iris, garantendo una convergenza rapida rispetto ad altri solver come `liblinear`.
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
{
  "alcohol": 13.5,
  "malic_acid": 2.1,
  "ash": 2.4,
  "alcalinity_of_ash": 18.0,
  "magnesium": 105,
  "total_phenols": 2.7,
  "flavanoids": 2.9,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.8,
  "color_intensity": 5.2,
  "hue": 1.0,
  "od280/od315_of_diluted_wines": 3.1,
  "proline": 950
}

### Esempio risposta:
{
  "prediction_logistic": "Class 0",
  "prediction_knn": "Class 0"
}

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
