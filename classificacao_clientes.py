import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class GrupoClassificacaoClientes():

    def __init__(self, caminhoImagens):

        self.caminhoImagens = caminhoImagens

    def pegando_dados(self, nomeArquivo):

        dadosClientes = pd.read_csv(nomeArquivo)

        dadosClientes = dadosClientes.drop('CUST_ID', axis= 1)
        dadosClientes = dadosClientes.dropna()
        dadosClientes = dadosClientes.reset_index(drop= True)

        self.dadosClientes = dadosClientes


    def descobrindo_numero_grupos(self):
        
        self.escalador = StandardScaler()

        self.dadosCLientesEscalados = self.escalador.fit_transform(self.dadosClientes)

        listaWCSS = []

        rangeGrupos = range(1, 50)

        for i in rangeGrupos:

            kmeans = KMeans(n_clusters=  i)
            kmeans.fit(self.dadosCLientesEscalados)
            listaWCSS.append(kmeans.inertia_)

        plt.plot(listaWCSS)
        plt.title('Achando o numero otimo de clusters')
        plt.xlabel('Cluesters')
        plt.ylabel('WCSS')
        plt.savefig(f'{self.caminhoImagens}\elbow_method.png')
        plt.close()


    def criando_grupos(self, numeroClusters):

        kmeans = KMeans(numeroClusters)
        kmeans.fit(self.dadosCLientesEscalados)
        self.grupos = kmeans.labels_

        centroDadosPorCluster = kmeans.cluster_centers_
        centroDadosPorCluster = self.escalador.inverse_transform(centroDadosPorCluster)
        centroDadosPorCluster = pd.DataFrame(data= centroDadosPorCluster, columns= [self.dadosClientes.columns])

        dadosClientesComGrupos = pd.concat([self.dadosClientes, pd.DataFrame({'grupo': self.grupos})], axis=1)

        for i in dadosClientesComGrupos.columns:
            
            plt.figure(figsize= (35,5))

            for j in range(numeroClusters):
                
                plt.subplot(1, numeroClusters, j+1)
                grupo =dadosClientesComGrupos[dadosClientesComGrupos['grupo'] == j]
                grupo[i].hist(bins= 20)
                plt.title(f'{i} \nGrupo {j}')

            plt.savefig(f'{self.caminhoImagens}\histograma_{i}.png')

        
    def verificando_kmeans_com_pca(self):

        pca = PCA(n_components=2)
        dadosClientesPCA = pca.fit_transform(self.dadosCLientesEscalados)
        dadosClientesPCA

        dfPCA = pd.DataFrame(data= dadosClientesPCA, columns= ['pca1', 'pca2'])
        dfPCA = pd.concat([dfPCA, pd.DataFrame({'grupo': self.grupos})], axis=1)

        plt.figure(figsize= (8,8))
        ax = sns.scatterplot(x='pca1', y='pca2', hue='grupo', data= dfPCA, palette=['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple', 'black', 'orange', 'navy'])
        plt.savefig(f'{self.caminhoImagens}\scatterplot_kmeans.png')

if __name__ == '__main__':

    clientes = GrupoClassificacaoClientes(caminhoImagens= r'C:\Users\Caio\Documents\dev\github\grupo_classificacao_clientes\imagens')

    clientes.pegando_dados(nomeArquivo= r'dados_clientes.csv')
    clientes.descobrindo_numero_grupos()
    clientes.criando_grupos(numeroClusters= 10)
    clientes.verificando_kmeans_com_pca()