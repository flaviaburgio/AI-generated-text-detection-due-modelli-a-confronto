# Classificazione AI vs. Human Text tramite Multi-Layer Perceptron (MLP)

Progetto di classificazione binaria del testo finalizzato a distinguere contenuti generati da modelli linguistici (**AI**, etichetta `1`) da testi scritti da esseri umani (**Human**, etichetta `0`), confrontando un approccio basato su rappresentazioni semantiche dense (**Sentence Embeddings**) con un approccio fondato su **feature stilometriche e linguistiche esplicite** mediante reti neurali **Multi-Layer Perceptron (`MLPClassifier`)**.

---

## Struttura del Dataset e Split

* **Dimensione complessiva:** 1.000 testi bilanciati (500 *Human* = `0`, 500 *AI* = `1`).
* **Suddivisione train / test:** rapporto `80/20` tramite `train_test_split` (**800** campioni di training, **200** campioni di test).
* **Riproducibilità:** `random_state=42` fissato per garantire lo stesso split e un confronto coerente tra le diverse configurazioni sperimentali.

---

## Metodologia e Approcci Sperimentali

### 1. Approccio 1 — Dense Semantic Embeddings
* **Rappresentazione:** estrazione di vettori densi a **384 dimensioni** tramite il modello `all-MiniLM-L6-v2` della libreria `sentence-transformers`.
* **Obiettivo:** valutare la capacità discriminativa di rappresentazioni semantiche compatte pre-addestrate nel separare testi umani e generati da AI.

### 2. Approccio 2 — Feature Linguistiche e Stilometriche
* **Rappresentazione:** estrazione di **6 feature** numeriche (matrice di forma `(1000, 6)`):
  * `readability_grade_level`
  * `lexical_diversity_ratio`
  * `burstiness_index`
  * `syntactic_variability`
  * `semantic_coherence_score`
  * `perplexity_score`
* **Pre-processing:** normalizzazione delle feature tramite `StandardScaler`.

### Configurazione del Modello (`MLPClassifier`)
* Architettura a singolo strato nascosto (`hidden_layer_sizes=(h,)`) valutata sistematicamente al variare del numero di neuroni `h` nell'insieme:
  `[20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]`
* Attivazione di `early_stopping=True` (con split di validazione interno gestito da `scikit-learn` sul set di training da 800 esempi).

---

## Sintesi dei Risultati

* **Superiorità degli Embeddings:** i vettori densi di `all-MiniLM-L6-v2` superano nettamente le feature stilometriche esplicite, raggiungendo picchi fino a **0.91** in *accuracy* e *F1-score*, rispetto a un massimo intorno a **0.78** ottenuto con le 6 feature linguistiche.
* **Dinamica del numero di neuroni e `early_stopping`:**
  * Con gli **embeddings**, l'interazione tra `early_stopping=True` e la dimensione contenuta del training set (800 istanze, ulteriormente ridotte dal validation split interno) produce oscillazioni tra diverse ampiezze dello strato nascosto (ad esempio `0.78` a 20 neuroni, `0.89` a 30, `0.82` a 40, `0.91` a 50).
  * Con le **feature linguistiche**, le prestazioni risultano più contenute ma relativamente stabili intorno al **0.74–0.78**.

1. Apri il notebook `.ipynb` in Jupyter Notebook o Google Colab.
2. Verifica che il dataset e le dipendenze (`scikit-learn`, `sentence-transformers`, `pandas`) siano installate e accessibili nel percorso previsto dal notebook.
3. Esegui le celle in ordine sequenziale per riprodurre l'estrazione delle feature/embeddings, l'addestramento degli `MLPClassifier` e il confronto delle metriche di valutazione.
