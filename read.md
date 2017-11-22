## README

### Tutorial

###### INDICE
1. [Prerequisiti](#pre)
2. [Creazione ambiente virtuale](#virtual-env)
3. [Installazione pacchetti](#req)
4. [Esecuzione degli script](#exec)
    * [PyCharm](#pycharm)
    * [Console](#console)
        - [Neural Network](#nn)
        - [Convolutional Neural Network](#cnn)
        - [Unsupervised Neural Network](#unn)

### Prerequisiti <a name="pre"></a>
Il presente tutorial è prodotto per il progetto OIL SPILL. Tutto è stato
sviluppato in ambiente linux usando la distro **Ubuntu 16.04** con il
supporto del **IDE PyCharm**. Si consiglia di utilizzare la
versione _pyhton3.5_.

La versione di pyhton indicata è presente di default nella versione del
sistema operativo usato, tuttavia, se si dovesse utlizzare un sistema
operativo diverso o meno recente, è possible installare la versione
indicata di pyhton direttamente dai sorgenti seguendo questa
[guida](https://passingcuriosity.com/2015/installing-python-from-source/)

**ATTENZIONE**

Prima di procedere all'installazione direttamente dai sorgenti
assicurarsi della piena compatibilità del proprio sistema operativo!


Affinché sia possibile eseguire i vari **_notebook jupyter_** di esempio
presenti nel progetto, è necessario aver installato la suite di **MATLAB**.

Per rendere il progetto indipendente dall'intero sistema operativo si
consiglia di intallare il gestore di ambiente virtuale. Tra i più usati
esiste _virtual_env_. Il suo utilizzo è fortemente consigliato.

### Creazione ambiente virtuale <a name="virtual-env"></a>
Dopo aver scaricato il codice sorgente del progetto è necessario
instanziare il proprio _virtual_env_.

Da questo punto in avanti assumiamo di essere in ambiente linux,
fermorestando che è possible effettuare le stesse operazioni in ambinete
windows.

Prima di procedere con la creazione del nostro ambiente virtuale è
necessario installare il pacchetto `python3-dev`

```
~$ sudo apt-get install python3-dev
```

Nel folder parent del progetto lanciare i seguenti comandi

```
~$ virtualenv -p python3.5 --distribute {nome_progetto}
~$ cd {nome_progetto}
~$ source bin/activate
```

A questo punto abbiamo creato ed attivato il nostro ambiente virtuale.

### Installazione pacchetti <a name="req"></a>
È necessario ora installare tutte le dipendenze del progetto, per farlo
lanciare il seguente comando:

```
~$ pip install -r requirements.txt
```

A questo punto il nostro ambiente virtuale è pronto per eseguire il
codice pyhton del progetto.

### Esecuzione degli script <a name="exec"></a>

Per poter eseguire il codice prodotto ci sono due modalità:

1. attraverso l'ambiente di sviluppo _PyCharm_
2. direttamente nella console eseguendo lo script `main.py`

##### PyCharm <a name="pycharm"></a>

Per lanciare l'esecuzione dello script dall'ambiente di sviluppo
consigliato è sufficiente lanaciare il run dello script `main.py` nella
cartella _src_ e seguire le istruzione nel terminale.

##### Console <a name="console"></a>

Per eseguire lo script nella console è necessario lanciare i seguenti
comandi nella console


```
~$ cd src
~$ python main.py




== MENU Classification OilSpill ==
----------------------------------
1 - Neural Network
2 - Convolutional Neural Network
3 - Fit and Evaluete Unsupervised Network

> _

```

###### Neural Network <a name="nn"></a>
Selezionando la voce **1** del menu si esegue la parte di script realativa
alle _**Neural Network**_

```

> 1
1 - Fit and Evaluete Neural Network
2 - Evaluete Neural Network
3 - back

>
```

Seguendo le voci del menu:

1. addestra, valuta la rete sul dataset completo  presente nel folder
`dataset`(utlizzando  la cross validation) e salva i modelli migliori
per i singoli `k_fold` e il _best_model_ dell'intera fase di training
2. consete di scegliere uno dei migliori modelli della fase di training
e usarlo per la fase di valutazione, al termine della quale salva il
modello usato nel folder `best_model` con i dati di _loss_ e _accuracy_
3. torna al menù precedente

###### Convolutional Neural Network <a name="cnn"></a>

Per eseguire l'addetramento o la valutazione della rete nurale
convoluzionale selezionare l'opzione **2** dal menù princiaple.

A questo punto bisogna selezionare l'opzione desiderata tra le seguenti

```

> 2
1 - Fit and Evaluete CNN
2 - Evaluete CNN
3 - back

>
```

Seguendo le voci del menu:

1. addestra, valuta la rete sul dataset completo sfruttando il nostro
modello base di rete nurale convoluzionale. Alla fine del processo di
training salva il modello appena preparato e ne valuta le performances.
2. consete di valutare le performances del modello pre-addestrato con il
dataset selezionato e ne stampa le metriche fondamentali
```
Confusion Matrix :
[[1432    6    0]
 [   6  106   14]
 [   0    0   18]]

    Metrics =>  ['loss', 'acc'] [0.040917391368975646, 0.98356510745891279]
    Classification Report :
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      1438
          1       0.95      0.84      0.89       126
          2       0.56      1.00      0.72        18

    avg / total       0.99      0.98      0.98      1582
    ```
3. torna al menù precedente

###### Unsupervised Neural Network <a name="unn"></a>

Scegliendo l'opzione **_3_** dal menu principale si esegue l'addestramento
e l'inferenza di un autoencoder per rigenerare un nuovo dataset a
partire da quello originale. Questo ha l'obiettivo di usare una nuova
base di informazioni ricollocate secondo quello che l'autoencoder
riconosce nella morfologia delle immagini.

```
> 3
STARTING FITTING UNSUPERVISED NEURAL NETWORK
loading data .........

```

Alla fine del processo avremo un nuovo folder, `img_cluster`, nel quale
saranno presenti i folder per le 3 classi che contengono le immaigni
orginali ridistribuite secondo la logica dell'autoencoder