## README

### Tutorial

###### INDICE
1. [Prerequisiti](#pre)
2. [Creazione ambiente virtuale](#virtual-env)
3. [Installazione pacchetti](#req)

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