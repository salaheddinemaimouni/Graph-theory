import numpy as np          # importation de numpy
from tkinter import *       # importation de tous les elements de tkinter
from tkinter.ttk import Notebook    # importation du Modele Notebook depuis tkinter.ttk
import networkx as nx
import matplotlib.pyplot as plt
import copy as copy
import math
from canavas_graph_drawer import *

"""test10=[[0,2,"inf",4,"inf","inf","inf","inf"],["inf",0,"inf",-1,4,5,"inf","inf"],[-3,"inf",0,"inf","inf","inf",1,"inf"],["inf","inf",2,0,"inf","inf","inf","inf"],["inf","inf","inf","inf",0,"inf","inf",2],["inf","inf","inf","inf","inf",0,3,"inf"],["inf","inf","inf",-2,"inf","inf",0,2],["inf","inf","inf","inf","inf",-3,"inf",0]] # OV BellmanF
#test0=[[0,2,inf,4,inf,inf,inf,inf],[inf,0,inf,-1,4,5,inf,inf],[-3,inf,0,inf,inf,inf,1,inf],[inf,inf,2,0,inf,inf,inf,inf],[inf,inf,inf,inf,0,inf,inf,2],[inf,inf,inf,inf,inf,0,3,inf],[inf,inf,inf,-2,inf,inf,0,2],[inf,inf,inf,inf,inf,-3,inf,0]] # OV Bell




# fonction dijkstra qui utiluse les lists  exemple: matrix = [[0,2,3],["inf",0,1],[3,"inf",2n'est pas utilusée dans le programme
    réalisé avant de connetre qu'il faut utiluser le module numpy"""
######################################
"""def Dijkstra(M,L,P,matrix):   # fonction dijkstra qui utiluse les listes et qui prend en argument la table M,L,P et la matrice adjacente 
    l,p,a,b,T=[],[],[],[],[]  # définition des lists intermidieres
    for i in range(len(L[0])): # boucle initial i va de somet 1 vers sommet n
        T=[]
        a=L[i].copy()
        b=P[i].copy()             # ici copiage 
        l.append(a)
        p.append(b) 
        for j in range(len(L[0])):   # boucle secondaire j va de sommet 1 vers sommet n et cherche le plus petit element dans sommet 
            if (j+1 in M[i]) or L[i][j]=="inf":     
                pass                        # si on a traité le sommet ou il contient une distance infinie entre le et le sommet début ne faire aucune chose
            else:
                T.append(L[i][j])      # si non ajouté sa distance à la liste T
        try:                            
            M.append(M[i]+[L[i].index(min(T))+1])   # essayer d'ajouter le sommet de plus bas distance à M 
        except:
            break                      # si on a traité tous les sommets, un erreur inerrompu le programme alors sortir de la boucle initial
        L.append(L[i])                # ajouter la ligne qu'on a traité aux tbleaux
        P.append(P[i])              
        for j in range(len(L[0])):        # boucle secondaire j va de somet 1 vers sommet n pour trouver le plus court chemain
            if j+1 in M[i+1]:           # si on a traité le sommet ne faire aucune chose
                pass
            else:               # si on n'a pas traité le sommet 
                if matrix[L[i].index(min(T))][j]=="inf":      
                    pass                                              # s'il y a une distance infinie à ajouter ne faire aucune chose
                elif L[i+1][j]=="inf":
                    L[i+1][j]=min(T)+matrix[L[i].index(min(T))][j]      # s'il y a une distance infinie à remplacer donc remplacer
                    P[i+1][j]=L[i].index(min(T))+1
                elif L[i+1][j]>min(T)+matrix[L[i].index(min(T))][j]:
                    L[i+1][j]=min(T)+matrix[L[i].index(min(T))][j]     # s'il y a une grande distance donc remplacer
                    P[i+1][j]=L[i].index(min(T))+1
        print(M,l,p)
        # refaire la boucle pour tous les sommets 
    return [M,l,p]         # renvoyer une list qui est le tableau complet de dijkstra"""






def dijkstra(matrix,S):
    M_tab, L_tab, P_tab,Ind = [], [], [],0  # 3 lists qu'on utilise pour retourner le tableau à afficher dans GUI
    # initialisation
    M = np.array([S])   # initialiser M comme un vecteur d'une seule dimention qui contien le point de début
    M_tab.append(M)     # ajouter M dans M_tab
    matrix[S][S]=0     # initialiser la petite longueur de S vers S comme 0
    L = np.array([matrix[S]])   # initialiser L comme la ligne de S dans la matrice adjacente
    L_tab.append(L[0])       # ajouter L dans L_tab
    p = []                 # p c'est une list intermidiaire pour initialiser P
    for e in L[0]:      # e est la distance entre S et les autres sommets
        if e<float("inf"):
            p.append(S)    # si e est finie alors S est le prédessesseur 
        else:
            p.append(None)   # si non S n'est pas un predecesseurs
    P = np.array(p)      # initialiser P 
    P_tab.append(P)     # ajouter P dans P_tab
    print(M,L,P)
    # traitement
    for i in range(len(L[0])):  # tant que M != V # boucle initial, i va de sommet 0 vers sommet n-1

        # coisir x C V\M tq pour tout y C V\M, L[x]<=L[y]
        Min_lst = []   # list intermidiaire qui va contenir le minimum pour chaque valeur de i
        for j in range(len(L[0])):  # j va de 0 à n-1
            if j in M:       
                pass     # si non a traité le sommet ne faire auccune chose  
            else:
                Min_lst.append(L[0][j])   #si non ajouter sa distance à Min_list
        if Min_lst == []:      
            break                # si Min_list est vide alor on a traité tout les sommet donc sortir de la boucle
        Min_lst = np.array(Min_lst)    # transformer Min_lst à un module numpy
        Min_val = Min_lst.min()        # affecter la distance minimal de Min_lst à Min_val
        var = np.where(L[0]==Min_val)
        try:
            Index_min = int(var[0])    # affecter l'index de la valeur minimal à Index_min s'il y'un un seul minimal
        except:
            for l in range(len(var[0])): # si non choisir le premier minimum
                if not(int(var[0][l]) in M):
                    Index_min = int(var[0][l])
                    break
                else:
                    pass
        M = np.append(M,[Index_min])       # ajouter l'index de la distance minimal au tableau M
        M_tab.append(M)             # ajouter M à M_tab
        # L_t1 tableau L temporisé "c'est les plus petites distances entre le sommet S et les autres sommet à travet le premier sommet choisi "
        L_t1 = matrix[Index_min]+Min_val    
        L_t2 = np.array([])       # L_t2 tableau L temporisé vide
        for j in range(len(L[0])):
            # ajouter à L_t2 la plus petite distance entre la distance déja trouvé et la nouvelle distance trouvé
            L_t2 = np.append(L_t2,[np.array([L[0][j],L_t1[j]]).min()])   
        L_tab.append(L_t2)    # ajouter L_t2 à L_tab 
        L = np.array([L_t2])   # affecter L_t2 à L
        P_t=np.array([])         # P_t tableau P temporisé
        for j in range(len(L[0])):
            if L_tab [i][j]==L_tab [i+1][j]:
                P_t = np.append(P_t,P_tab[i][j])   # si on n'a pas modifié la distance donc reste le meme predecesseur
            else:
                P_t = np.append(P_t,[Index_min])   # si non le predecesseur sera Index_min 
        P_tab.append(P_t)             # ajouter P_t à P_tab
        print(M_tab[-1],"M",L_tab[-1],"L",P_tab[-1],"P")        # afficher le tableau de dijkstra

    return [M_tab,L_tab,P_tab]       # renvoyer une list qui est le tableau complet de dijkstra



iteration=0
# fonction bellmanford qui utilise les lists exemple: matrix = [[0,2,3],["inf",0,1],[3,"inf",2]]
# fonction bellmanford qui prend L,P,matrix comme arguments + sortie1,sortie2 initialiser par des lists vides pour les renvoyer à la fin des itérations
def bellmanford(L,P,matrix,sortie1,sortie2):
    print(L,P)
    T,l=[],[]           # lists intermidiaires
    global iteration
    l=L[0].copy()
    for i in range(len(matrix)):  # boucle initial
        #print("BF",i,"   ",iteration,np.array(L[-1]),np.array(P[-1]))   # afficher la table de bellman ford
        L.append([])
        P.append([]) # ajouter des lists vids à L et P pour les remplis 
        for j in range(len(matrix)): # boucle secondaire
            if L[i][i]=="inf":   # si la distance L[i][i] est infinie ajouter L P de [i] et L P de [i+1] restent les memes puis sortir de la boucle
                L[i+1]=L[i]
                P[i+1]=P[i]
                break
            elif L[i][j]=="inf" or (matrix[i][j]!="inf" and L[i][j]>matrix[i][j]+L[i][i]):  #si la distance est grande
                if matrix[i][j]=="inf":  # si il est infinie ajouter à L[i+1] la distance entre le sommet i et le sommet j et à P[i+1] None
                    L[i+1].append(matrix[i][j])
                    P[i+1].append(None)
                else:   # si non ajouter à L[i+1] la distance entre le sommet i et le sommet j + la distance L[i][i]
                    L[i+1].append(matrix[i][j]+L[i][i])
                    if matrix[i][j]+L[i][i]<float("inf"):
                        P[i+1].append(i+1)
                    else:
                        P[i+1].append(None)
            else:   # si non ajouter à L[i+1] la distance entre le sommet i et le sommet j et à P[i+1] le predecesseur qui est déja dans P[j][j]
                L[i+1].append(L[i][j])
                P[i+1].append(P[i][j])
    if all(L[i]==l for i in range(len(matrix))): # si non a terminé une itération sans modification donc arreter
        iteration = 0  # reset iteration 
        return [sortie1,sortie2]
    else:             # si non refaire la meme fonction
        sortie1.append(L)
        sortie2.append(P)
        iteration +=1   # incrementer le nombre d'iteration
        return bellmanford([L[-1]],[P[-1]],matrix,sortie1,sortie2)










    
    
# fonctions

def ttm(txt,logarithm):  # fonction qui converti la matrice adjacente à une list
    L=txt.split("\n")   # list de str
    l,m=[],[]
    for i in range(len(L)):
        if L[i]=="" or L[i]==" " or L[i]=="  " or L[i]=="   " or L[i]=="    "  :
            pass   # si il ya un espace ne faire aucun chose
        else:
            l.append(L[i].split(","))  # si non ajouter l'element de L à l
    for e,i in zip(l,range(len(l))):
        m.append([])
        for f in e:
            try:
                if f=="inf":
                    m[i].append(float('inf')) # s'il ya aucun problem ajouter la valeur Matrix[ligne][colone] à m
                else:
                    if logarithm==0:        # si on a la somme entre les sommet
                        m[i].append(float(f))
                    else:                   # si on a le produit utiluser la fonction logarithme neperienne
                        if float(f)>0:
                            m[i].append(round(math.log(float(f)),3))
                        else:    # s'il ya un nombre <=0 le log n'est pas définit
                            return "log ne marche pas avec les 0 ou les nombres negatifs"
            except:
                return "erreur de syntax"  # s'il ya une erreur renvoyer "erreur de syntax"
    if all(len(n)==len(m) for n in m):
        return m  # si la matrice est carée la retourner 
    else:
        return "la matrice n'est pas carée" # si non renvoyer "la matrice n'est pas carée"

def S_detect(m):  # fonction qui detecte le graph s'il est symetrique ou non
    for i in range(len(m)):
        for j in range(len(m)):
            
            if m[i][j]==m[j][i]: #si les element symetrique sont égaut pass
                pass
            else:   # si non le graph est non symetrique donc renvoi False
                return False
    return True    # si la boucle termine sans renvoyer False donc il est symetrique alors renvoi True

def V_detect(m):  # fonction qui detecte le graph s'il est symetrique ou non
    for i in range(len(m)):
        for j in range(len(m)): 
            if m[i][j]==0 or m[i][j]==1 :  #si le'element est 1 ou 0 pass 
                pass
            else:
                return True   # si non le graph est valué donc renvoi True
    return False   # si la boucle termine sans trouvé une valeur different de 1 ou 0 alros c'est t'un graph non valué alors renvoi False


def dictodic(D): # une fonction qui adabte une liste contien des listes dont le preimer element est le debut et les autes sont des successeurs ou predecesseurs à une liste qui contient des tuples pour les afficher dans le tableau quand on clique sur le bouton dictionnaire
    lst = []
    for e in D:
        char=",".join(str(e[i]) for i in range(1,len(e)))
        lst.append((e[0],char))   
    return lst     # retourner la list souhaité

        







# classes graphs

class Graph:    # class Graph
    def __init__(self,matrix):
        self.matrix = np.array(matrix)  # variable matrice adjacente comme module de numpy

    def Ordre(self):  # methode qui renvoi l'ordre de graph
        return len(self.matrix)


class GraphNONV(Graph):   # class graph non orienté non valué
    
    def init(self,matrix):          # initialisation
        self.__init__(Graph)
        
    """def Taille(self):               # methode qui determine la taille du graph
        IntermideaireVar,T=0,0
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if i!=j and self.matrix[i][j]==1:
                    IntermideaireVar = IntermideaireVar + 1 # si les element non diagoneaux egale à 1 donc incrementer la variable
                elif i==j and self.matrix[i][j]==1:
                    T = T + 1                     # si il y'a une boucle dans un sommet incrementer la variable T
        return int(T + (IntermideaireVar/2))    # la taille eqale à T + (IntermideaireVar/2)"""

    def dic(self):      # methode qui retourn une list tq le preimer element est le debut et les autes sont des successeurs ou predecesseurs
        L=[]
        for i in range(len(self.matrix)):
            L.append([i])
            for j in range(len(self.matrix)):
                if i==j:     # si nous somme dans la diagonal ne faire aucune chose 
                    pass
                elif self.matrix[i][j]==1: # si l'element non diagonal egale à 1 ajouter le sucssesseur ou le predecesseur
                    L[i].append(j)    
        return L    # renvoyer le dictionnaire comme liste à fin de l'afficher dans le GUI

    def degre(self,A):   # methode qui retourne le degrée du somet A
        D=0
        for e in self.matrix[A]:
            if e==1:
                D=D+1   # le degrée est egale aux nombre des 1 dans la ligne corespendante à A
        print(A,"son degré",D)
        return D

    def E(self):   # methode qui test si le graph est eulerien ou non
        D=[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))
        if all(d%2==0 for d in D):
            print("le graph est eulerien")
            return True  # si tous les degrée sont paire donc True
        else:
            print("le graph n'est pas eulerien")
            return False   # si non False

    def SE(self):   # methode qui test le graph s'il est semi eulerien
        D,X,impaire,extrimite=[],[],0,[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))   # ajouter à la list D les degrée de chaque somet
        for d in D:
            X.append(d%2)      # ajouter à la liste X les restes de division de chaque degrée
        for x,i in zip(X,range(len(X))):
            if x==1:              
                extrimite.append(i+1)
                impaire=impaire+1
        if impaire==2:
            print("semi eulerien avec",extrimite)
            return extrimite   # s'il ya exactement deux degrée impaire et les autres paires donc le graph est semi eulerien
        else:
            print("n'est pas semi eulerien")
            return False      # si non le graph n'est pas semi eulerien
              
                    
                
                
        
        

        

class GraphONV(Graph):    # class graph orienté non valué
    def init(self,matrix):  # initialisation
        self.__init__(Graph)

    """def Taille(self):     
        T=0
        for ligne in self.matrix:
            for e in ligne:
                if e==1:
                    T = T + 1
        return T"""

    def Sdic(self):     # methode qui retourn une list tq le preimer element est le debut et les autes sont des successeurs 
        L=[]
        for i in range(len(self.matrix)):
            L.append([i])
            for j in range(len(self.matrix)):
                if i==j:
                    pass     # si nous somme dans la diagonal ne faire aucune chose 
                elif self.matrix[i][j]==1:  # si l'element non diagonal egale à 1 ajouter le sucssesseur 
                    L[i].append(j)
        return L       # renvoyer le dictionnaire comme liste à fin de l'afficher dans le GUI
    
    def Pdic(self):      # une list tq le preimer element est le debut et les autes sont des predecesseurs
        Tmatrix= self.matrix.T
        L=[]
        for i in range(len(Tmatrix)):
            L.append([i])
            for j in range(len(Tmatrix)):
                if i==j:
                    pass   # si nous somme dans la diagonal ne faire aucune chose 
                elif Tmatrix[i][j]==1:
                    L[i].append(j)   # si l'element non diagonal egale à 1 ajouter le predecesseur
        return L         # renvoyer le dictionnaire comme liste à fin de l'afficher dans le GUI

    def degreS(self,A):     # methode qui renvoi le degrée sortant d'un sommet
        D=0
        for e in self.matrix[A]:
            if e==1:
                D=D+1   # si l'element dans la ligne de A egale à 1 donc incrementer le degrée sortant
        return D   # renvoyer le degrée sortant
    
    def degreE(self,A):       # methode qui renvoi le degrée sortant d'un sommet
        D=0
        Tmatrix=self.matrix.T
        for e in Tmatrix[A]:
            if e==1:
                D=D+1  # si l'element dans la colone de A egale à 1 donc incrementer le degrée sortant
        return D   # renvoyer le degrée sortant
    
    def degre(self,A):
        return self.degreE(A)+self.degreS(A)    # methode qui renvoi le degrée totale du sommet A

    """def E(self):     
        D=[]
        for i in range(len(self.matrix)z):
            D.append(self.degre(i))
        print(D)
        if all(d%2==0 for d in D): return True
        else: return False

    def SE(self):
        D,X,impaire,extrimite=[],[],0,[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))
        for d in D:
            X.append(d%2)
        for x,i in zip(X,range(len(X))):
            if x==1:
                extrimite.append(i+1)
                impaire=impaire+1
        if impaire==2: return extrimite
        else: return False"""


        

class GraphNOV(Graph): # class graph non orienté valué
    def init(self,matrix):  #initialisation de la classe
        self.__init__(Graph)

    """def Taille(self):
        IntermideaireVar,T=0,0
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if i!=j and self.matrix[i][j]!=0 and self.matrix[i][j]!="inf":
                    IntermideaireVar = IntermideaireVar + 1
                elif i==j and self.matrix[i][j]!=0 and self.matrix[i][j]!="inf":
                    T = T + 1
        return int(T + (IntermideaireVar/2))"""

    def dic(self):    # methode qui retourn une list tq le preimer element est le debut et les autes sont des successeurs ou predecesseurs
        L=[]
        for i in range(len(self.matrix)):
            L.append([i])
            for j in range(len(self.matrix)):
                if i==j:
                    pass       # si nous somme dans la diagonal ne faire aucune chose 
                elif self.matrix[i][j]!=0 and self.matrix[i][j]!=float("inf") :
                    L[i].append(j)   # si l'element non diagonal egale à 1 ajouter le sucssesseur ou le predecesseur
        return L     # renvoyer le dictionnaire comme liste à fin de l'afficher dans le GUI

    def degre(self,A):       # methode qui renvoi le degrée sortant d'un sommet
        D=0
        for e in self.matrix[A]:
            if e!=0 and e!=float("inf"):
                # si l'element dans la colone de A egale à un nombre non nulle et finie donc incrementer le degrée sortant
                D=D+1    
        #print(A,"son degré",D)
        return D   # renvoyer le degrée sortant
    
    def E(self):     # methode qui verifie si le graph est eulerien ou non 
        D=[] 
        for i in range(len(self.matrix)):
            D.append(self.degre(i))     # ajouter à D les degrées de chaque sommet
        if all(d%2==0 for d in D): return True    # s'ils sont tous paires donc eulirien
        else: return False    # si non n'est pas eulerien

    def SE(self):      # methode qui verifie si le graph est semi eulerien ou non 
        D,X,impaire,extrimite=[],[],0,[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))    # ajouter à D les degrées de chaque sommet
        for d in D:
            X.append(d%2)     # ajouter à X le reste de division des degrée par 2
        for x,i in zip(X,range(len(X))):
            if x==1:
                extrimite.append(i+1)
                impaire=impaire+1
        if impaire==2: return extrimite   # s'il y'a exactement deux sommets de degrée impaire et les autres paires donc semi eulerien et renvoyer les extrimité
        else: return False    # si non n'est pas semi eulerien

    """def DH(self):
        D=[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))
        if all(d>=(len(D)/2) for d in D): return True
        else: return False"""


    """def Dijkstra(self,A):
        mat=self.matrix.tolist()
        mta=[]
        for e in mat[0]:
            if e=="inf":
                mta.append("inf")
            else:
                mta.append(float(e))
        M=[[A]]
        L=[mta]
        P=[[]]
        for e,i in zip(L[0],range(len(self.matrix))):
            if i+1==A or (e!=0 and e!="inf"):
                P[0].append(A)
            elif e=="inf":
                P[0].append(None)
        X=[]
        for e,i in zip(self.matrix,range(len(self.matrix))):
            X.append([])
            for f in e:
                if f==float('inf'):
                    X[i].append("inf")
                else:
                    X[i].append(float(f))
        return dijkstra(M,L,P,X)"""
    
    def dijkstra(self,S):    # methode qui utiluse la fonction dijkstra écrit dans le début de ce code
        M = []
        for i in range(len(self.matrix)):
            M.append([])
            for j in range(len(self.matrix)):
                M[i].append(float(self.matrix[i][j])) # ajouter à la liste M les elements de la matrice adjacente
        M = np.array(M)    # transformer M à une matrice de module numpy
        return dijkstra(M,S)    # renvoyer la fonction dijkstra(matrix,S)
  
    def BellmanFord(self,A):    # methode qui utiluse la fonction Bellman Ford écrit dans le début de ce code
        L=[["inf"]*(A-1)+[0]+["inf"]*(len(self.matrix)-A)]   # initialiser  la table L
        P=[[None]*(A-1)+[A]+[None]*(len(self.matrix)-A)]   # initialiser  la table P
        return bellmanford(L,P,self.matrix,[],[])      # renvoyer la fonction Bellman_ford(L,P,Matrix,[],[])

    def FloydWarshall(self): # mthode qui execute l'algorithme de Floyd Warshall
        n = len(self.matrix)   
        M = copy.copy(self.matrix)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # remplacer par la plus petite distance trouvé entre le sommet j et le sommet k
                    M[j][k] = min(M[j][k], M[j][i] + M[i][k])
        print(M)
        return M     #renvoyer la matrice de foyd warshall



class GraphOV(Graph):   # class graph orienté valué
    def init(self,matrix):   # initialisation
        self.__init__(Graph)

    """def Taille(self):
        T=0
        for ligne in self.matrix:
            for e in ligne:
                if e!="inf":
                    T = T + 1
        return T"""

    def Sdic(self):     # methode qui retourn une list tq le preimer element est le debut et les autes sont des successeurs 
        L=[]
        for i in range(len(self.matrix)):
            L.append([i])
            for j in range(len(self.matrix)):
                if i==j:
                    pass  # si nous somme dans la diagonale ne faire aucune chose 
                elif self.matrix[i][j]!=0 and self.matrix[i][j]!=float("inf") :  
                    L[i].append(j)   # si l'element n'est pas nulle et finie donc il est un successeur
        return L


    def Pdic(self):      # une list tq le preimer element est le debut et les autes sont des predecesseurs
        Tmatrix= self.matrix.T  # maintenant on travail avec la matrice transposé
        L=[]
        for i in range(len(Tmatrix)):
            L.append([i])
            for j in range(len(Tmatrix)):
                if i==j:
                    pass   # si nous somme dans la diagonale ne faire aucune chose 
                elif Tmatrix[i][j]!=0 and Tmatrix[i][j]!=float("inf") :
                    L[i].append(j)   # si l'element n'est pas nulle et finie donc il est un successeur
        return L

    def degreS(self,A):   # methode qui renvoi les degrée sortant du sommet A 
        D=0
        for e in self.matrix[A]:
            if e!=0 and e!=float('inf'):
                D=D+1   # si l'element non nulle et finie donc incrementer le degrée
        return D    
    
    def degreE(self,A):   # methode qui renvoi les degrée entrant du sommet A 
        D=0
        for e in self.matrix.T[A]: # maintenant on travail avec la matrice transposé
            if e!=0 and e!=float('inf'):
                D=D+1    # si l'element non nulle et finie donc incrementer le degrée
        return D


    def degre(self,A):    # methode qui renvoi le degrée totale du sommet A
        print(A,"son degré",self.degreE(A)+self.degreS(A))
        return self.degreE(A)+self.degreS(A)

    """def E(self):
        D=[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))
        if all(d%2==0 for d in D): return True
        else: return False

    def SE(self):
        D,X,impaire,extrimite=[],[],0,[]
        for i in range(len(self.matrix)):
            D.append(self.degre(i))
        for d in D:
            X.append(d%2)
        for x,i in zip(X,range(len(X))):
            if x==1:
                extrimite.append(i+1)
                impaire=impaire+1
        if impaire==2: return extrimite
        else: return False"""

    """def Dijkstra(self,A):
        mat=self.matrix.tolist()
        mta=[]
        for e in mat[0]:
            if e=="inf":
                mta.append("inf")
            else:
                mta.append(float(e))
        M=[[A]]
        L=[mta]
        P=[[]]
        for e,i in zip(L[0],range(len(self.matrix))):
            if i+1==A or (e!=0 and e!="inf"):
                P[0].append(A)
            elif e=="inf":
                P[0].append(None)
        X=[]
        for e,i in zip(self.matrix,range(len(self.matrix))):
            X.append([])
            for f in e:
                if f==float('inf'):
                    X[i].append("inf")
                else:
                    X[i].append(float(f))
        return Dijkstra(M,L,P,X)"""

    ## meme commantaires que la classe précedente
    ## meme commantaires que la classe précedente
    ## meme commantaires que la classe précedente 

    def dijkstra(self,S):
        print(self.matrix)
        M = []
        for i in range(len(self.matrix)):
            M.append([])
            for j in range(len(self.matrix)):
                M[i].append(float(self.matrix[i][j]))
        M = np.array(M)
        return dijkstra(M,S)

    def BellmanFord(self,A):
        L=[["inf"]*(A-1)+[0]+["inf"]*(len(self.matrix)-A)]
        P=[[None]*(A-1)+[A]+[None]*(len(self.matrix)-A)]
        return bellmanford(L,P,self.matrix,[],[])

    def FloydWarshall(self):
        print(self.matrix)
        n = len(self.matrix)
        M = copy.copy(self.matrix)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    M[j][k] = min(M[j][k], M[j][i] + M[i][k])
        return M










#tkinter GUI

class Main(Tk):            #le GUI comme une class de tkinter
    def __init__(self,fig=1):        
        super().__init__()        # initialisation de super 
        self.fig = fig    # variable qui s'incremente à chaque fois on ouvre une figure par matplotlib
        self.logvar = IntVar()    # variale qui prend 0 ou 1 selon l'état du checkbox d'activation de log
        self.divar = IntVar()
        self.valvar = IntVar()
        self.plchvar = IntVar()   # variale qui prend 0 ou 1 selon l'état du checkbox d'activation de plus long chemain pour l'algorithm de bellman ford
        self.title("Recherche operationnelle")            # le titre du GUI
        self.geometry("600x800")   # la geometry gu GUI


        def donothing():
            x = 0
        # le menu bar
        self.menubar = Menu(self)
        self.filemenu = Menu(self.menubar, tearoff=0)  # menu de fichers

        self.filemenu.add_command(label="Ouvrir le fichier matrice.txt", command=self.open_fun)   # bouton qui ouvre le fichier matrice.txe
        self.filemenu.add_command(label="Enregistrer la matrice dans matrice.txt", command=self.save_fun)   # boutton qui enregistre le fichier matrice.txe
        
        self.filemenu.add_separator()  # ajouter un separateur
        
        self.filemenu.add_command(label="Reset", command=self.reset_fun) # bouton qui fait le reset de grogramme
        
        self.menubar.add_cascade(label="fichier", menu=self.filemenu) # ajouter en cascad le menu fichier
        
        self.helpmenu = Menu(self.menubar, tearoff=0)   # menu Aide
        
        self.helpmenu.add_command(label="Help", command=donothing)  # bouton help 
        self.helpmenu.add_command(label="A propos", command=self.about_fun)   # bouton à propos
        
        self.menubar.add_cascade(label="Aide", menu=self.helpmenu)  # ajouter en cascad le menu Aide
        
        self.config(menu=self.menubar)   # configuration du menu bar

        #notebook 
        self.nb = Notebook(self)    # definir nb commme un Notebook

        # graph
        graph_tab = Frame(self.nb)    #fenetre des graphs

        self.button_height = 1
        # premier label
        self.graph_first_label = Label(graph_tab)
        self.graph_first_label ["text"] ="Ecrire la matrice d'adjacence, entrant +∞ = inf"
        self.graph_first_label.pack(side=TOP)

        # entrer de la matrice
        self.graph_text = Text(graph_tab)  # entrée du matrice adjacente
        self.graph_text ["width"] =50
        self.graph_text ["height"] =15
        self.graph_text.pack(side=TOP)   # affichage d'entrée du matrice adjacente

        self.draw_val = Checkbutton(graph_tab, text="graphe valué", variable=self.valvar)
        self.draw_di = Checkbutton(graph_tab, text="graphe orienté", variable=self.divar)
        self.draw_val.pack(side=TOP)
        self.draw_di.pack(side=TOP)
        
        # boutton qui ouvre la fenetre de traçage du graph
        self.draw_graph = Button(graph_tab)
        self.draw_graph ["text"] ="cliquer pour dessiner un graph"
        self.draw_graph ["width"] =100
        self.draw_graph ["height"] =self.button_height
        self.draw_graph ["command"] = self.draw_gaph_fun
        self.draw_graph.pack(side=TOP)
        
        # label des alerts
        self.graph_alert = Label(graph_tab)
        self.graph_alert ["text"] ="Alert"
        self.graph_alert ["bg"] ="WHITE"
        self.graph_alert.pack(side=TOP)

        # checkbutton de logarithm
        self.log = Checkbutton(graph_tab, text="Produit -> Utiluser la fonction ln(matrice[i][j])", variable=self.logvar)
        self.log.pack(side=TOP)

        # boutton qui affiche l'ordre et le type du graph
        self.execute_graph = Button(graph_tab)
        self.execute_graph ["text"] ="Cliquer pour voir l'ordre et le type du graph et activer les dictionnaires "
        self.execute_graph ["width"] =100
        self.execute_graph ["height"] =self.button_height
        self.execute_graph ["command"] = self.excute_gaph_fun
        self.execute_graph.pack(side=BOTTOM)

        # sortie d'ordre du graph
        self.graph_ordre = Label(graph_tab)
        self.graph_ordre ["text"] ="ordre"
        self.graph_ordre.pack(side=TOP)

        # sortie de type du graph
        self.graph_type = Label(graph_tab)
        self.graph_type ["text"] ="Type"
        self.graph_type.pack(side=TOP)

        # boutton qui affiche le dictionnaire des successeurs
        self.graph_Sdic = Button(graph_tab)
        self.graph_Sdic ["text"] ="Dictionnaire des successeurs"
        self.graph_Sdic ["width"] =100
        self.graph_Sdic ["height"] =self.button_height
        self.graph_Sdic.pack(side=BOTTOM)

        # boutton qui affiche le dictionnaire des predecesseurs
        self.graph_Pdic = Button(graph_tab)
        self.graph_Pdic ["text"] ="Dictionnaire des predecesseurs"
        self.graph_Pdic ["width"] =100
        self.graph_Pdic ["height"] =self.button_height
        self.graph_Pdic.pack(side=BOTTOM)

        # boutton qui afficher le graph utilusant matplotlib
        self.graph_plot = Button(graph_tab)
        self.graph_plot ["text"] =" Afficher le graphe"
        self.graph_plot ["width"] =100
        self.graph_plot ["height"] =self.button_height
        self.graph_plot.pack(side=BOTTOM)
        self.graph_plot ["command"] = self.gaph_plot_fun

        # boutton qui affiche les degrée de chaque sommet aussi qui l'eulerianité du graph
        self.graph_degres = Button(graph_tab)
        self.graph_degres ["text"] ="Degres"
        self.graph_degres ["width"] =100
        self.graph_degres ["height"] =self.button_height
        self.graph_degres.pack(side=BOTTOM)
        self.graph_degres ["command"] = self.gaph_degres_fun

        # boutton qui affiche le tableau de dijksta depuis un sommet sonnée
        self.graph_dijkstra = Button(graph_tab)
        self.graph_dijkstra ["text"] ="Dijkstra"
        self.graph_dijkstra ["width"] =100
        self.graph_dijkstra ["height"] =self.button_height
        self.graph_dijkstra.pack(side=BOTTOM)
        self.graph_dijkstra ["command"] = self.gaph_dijkstra_fun
        self.execute_graph = Button(graph_tab)

        # boutton qui affiche le tableau de bellmanford depuis un sommet donnée
        self.graph_BF = Button(graph_tab)
        self.graph_BF ["text"] ="Bellman Ford et Arborescence"
        self.graph_BF ["width"] =100
        self.graph_BF ["height"] =self.button_height
        self.graph_BF.pack(side=BOTTOM)
        self.graph_BF ["command"] = self.gaph_BF_fun

        # checkbox qui modifie l'algorithme bellman ford pour calculer le plus long chemain
        self.plch = Checkbutton(graph_tab, text="Utiluser Belmman ford pour plus long chemain", variable=self.plchvar)
        self.plch.pack(side=BOTTOM)

        # boutton qui affiche la matrice d'algorithme floyd warshall
        self.graph_FW = Button(graph_tab)
        self.graph_FW ["text"] ="Floyd Warshall"
        self.graph_FW ["width"] =100
        self.graph_FW ["height"] =self.button_height
        self.graph_FW.pack(side=BOTTOM)
        self.graph_FW ["command"] = self.gaph_FW_fun

        # text pour entrer le sommet de début pour dijkstra et bellmanford
        self.graph_S = Text(graph_tab)  
        self.graph_S ["width"] =5
        self.graph_S ["height"] =2
        self.graph_S.pack(side=BOTTOM)

        # sortie de eulerianité
        self.graph_E = Label(graph_tab)
        self.graph_E ["text"] ="???Eulerien??? \n Cliquer degres pour savoir"
        self.graph_E.pack(side=TOP)
        
        #simplex  à develloper plus tard
        simplex_tab = Frame(self.nb)   # fenetre de Simplex
        
        self.simplex_text = Text(simplex_tab)  # entrée du programme lineaire
        self.simplex_text.pack(side=TOP)   # affichage d'entrée de simplex

        # Notebook
        self.nb.add(graph_tab, text="Graphs")   # ajouter le frame du graph 
        #self.nb.add(simplex_tab, text="Simplex") # ajouter frame de simplex

        self.nb.pack()   # ajouter le notebook



    def reset_fun(self):  # commande de boutton reset
        self.plchvar.set(0)  # mettre bellman ford à plus cout chemain
        self.logvar.set(0)    # n'utiluse pas le log
        self.draw_val.set(0)
        self.draw_di.set(0)
        self.graph_text.delete(1.0,END)  # vider le text

    def save_fun(self):  # commande de boutton enregistrer
        if type(ttm(self.graph_text.get(1.0,END),self.logvar.get()))==str:  # s'il y'a un probleme dand la matrice
            # afficher l'alert s'il y'a un probleme
            self.graph_alert ["text"] = ttm(self.graph_text.get(1.0,END),self.logvar.get())
            self.graph_alert ["bg"] ="RED"
        else:
            # si non ouvrir le fichier matrice.txt
            with open('matrice.txt', 'w') as f:
                f.write(self.graph_text.get(1.0,END))
            # afficher que le fichier a était enregistré
            self.graph_alert ["text"] =" Fichier enregistrée"
            self.graph_alert ["bg"] ="YELLOW"

    def open_fun(self):    # commande de boutton ouvrir
        file = open('matrice.txt').read()  # ouvrir le fichier matrice.txt
        self.graph_text.delete(1.0,END)   # vider l'entrer du matrice
        self.graph_text.insert(0.0, file)  # ajouter le contenu de matrice.txt à l'entrée
        
    def about_fun(self):   # commande de boutton à propos
        about= Toplevel(self)  # pop up une fenetre
        about.title("à propos")  # le titre est à proppos
        # ecrir dans la fenetre le suivant
        Label(about, text= "Logiciel crée par \n SALAH EDDINE MAIMOUNI \n HAMZA CHAIB et MOHAMED ES-SALHY \n version 1.0", font=('Mistral 18 bold')).pack(side=BOTTOM) 


    def draw_gaph_fun(self):
        """choose = Toplevel(self)  # pop up une fenetre
        choose.geometry("330x100")
        choose.title("choisir le type de graph")  # le titre
        valvar = IntVar()
        divar = IntVar()
        btn_val = Checkbutton(choose, text="valué", variable=valvar)
        btn_val.pack()
        btn_di = Checkbutton(choose, text="orienté", variable=divar)
        btn_di.pack()"""
        draw = Tk()
        titre = "traçage du grap:"
        if self.valvar.get()==0:
            v = False
            titre = titre + " non valué"
        else:
            v = True
            titre = titre + " valué"
        if self.divar.get() == 0:
            d = False
            titre = titre + ", non orienté"
        else:
            d = True
            titre = titre + ", orienté"
        draw.title(titre)
        root = Draw_Graph(draw,directed=d,valued=v)
        draw.mainloop()
        
        """btn_choix = Button(choose, text= "choisir", command= get_v_d())
        btn_choix.pack()"""
            
        
    def excute_gaph_fun(self):   # commande de la bouton qui affiche le type et l'ordre du graph
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        try:   # s'il n'ya pas un erreur
            if type(A)==str:   # si A est un texte l'afficher dans l'alert
                self.graph_alert ["text"] = A
                self.graph_alert ["bg"] ="RED"
                self.graph_ordre ["text"] = ""
            else:    # si non

                # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
                S_A , V_A = S_detect(A) , V_detect(A)
                
                if S_A == True and V_A == True:
                    # graph Non Orienté ,Valué
                    G=GraphNOV(A)
                    self.graph_type ["text"] = "Non Orienté ,Valué"
                    self.graph_Sdic ["command"] = self.gaph_dic_fun
                    self.graph_Pdic ["command"] = self.gaph_dic_fun
                        
                elif S_A == True and V_A == False:
                    # graph Non Orienté ,Non Valué
                    G=GraphNONV(A)
                    self.graph_type ["text"] = "Non Oorienté Non Valué"
                    self.graph_Sdic ["command"] = self.gaph_dic_fun
                    self.graph_Pdic ["command"] = self.gaph_dic_fun
                    
                elif S_A == False and V_A == True:
                    # graph Orienté ,Valué
                    G=GraphOV(A)
                    self.graph_type ["text"] = "Orienté ,Valué"
                    self.graph_Sdic ["command"] = self.gaph_Sdic_fun
                    self.graph_Pdic ["command"] = self.gaph_Pdic_fun
                    
                elif S_A == False and V_A == False:
                    # graph Orienté , Non Valué
                    G=GraphONV(A)
                    self.graph_type ["text"] = "Orienté , Non Valué"
                    self.graph_Sdic ["command"] = self.gaph_Sdic_fun
                    self.graph_Pdic ["command"] = self.gaph_Pdic_fun

                # ecrir l'ordre de graph dans la sortie
                self.graph_ordre ["text"] = G.Ordre()
        except:   # s'il y'a un problame 
            self.graph_alert ["text"] = "error de syntaxte"
            self.graph_ordre ["text"] = ""



            
            
    def gaph_dic_fun(self):  # commande des boutton de successeurs et de predecesseurs si le graph est symetrique 
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:  # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == True and V_A == True:
            G=GraphNOV(A)    # graph Non Orienté , Valué
        elif S_A == True and V_A == False:
            G=GraphNONV(A)   # graph Non Orienté ,None Valué
        class Table:  # class tableau
            def __init__(self,root):  # initialisation
                for i in range(total_rows):
                    for j in range(total_columns):
                        # ajouter la sortie du methode dic dans la table qui va pop up
                        self.e = Entry(root, width=20, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst = dictodic(G.dic())   # lst est la diste qui contien le tableau de dictionnaire
        total_rows = len(lst)       # le nombre des ligne
        total_columns = len(lst[0])  # le nombre des colones
        # afficher la fenetre qui contien le tableu de dictionnaire graph non orienté
        root = Tk()        
        root.title("Dictionnaire graph non orienté")  # titre de fenetre 
        t = Table(root)
        root.mainloop()# afficher le tableau

    def gaph_Sdic_fun(self):  # commande de boutton de successeurs 
        # initialiser le label alert
        self.graph_alert ["text"] ="Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)   # graph orienté valué
        elif S_A == False and V_A == False:
            G=GraphONV(A)  # graph orienté non valué
        class Table:  # class tableau
            def __init__(self,root):  # initialisation
                for i in range(total_rows):
                    for j in range(total_columns):
                        # ajouter la sortie du methode Sdic dans la table qui va pop up
                        self.e = Entry(root, width=20, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst = dictodic(G.Sdic())   # lst est la diste qui contien le tableau de dictionnaire
        total_rows = len(lst)       # le nombre des ligne
        total_columns = len(lst[0])  # le nombre des colones
        # afficher la fenetre qui contien le tableu de dictionnaire des sucesseurs
        root = Tk()
        root.title("Dictionnaire des sucesseurs")   # titre
        t = Table(root)
        root.mainloop() # afficher le tableau

    def gaph_Pdic_fun(self):    # commande de boutton de predecesseurs
        # initialiser le label alert
        self.graph_alert ["text"] ="Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)  # graph orienté valué
        elif S_A == False and V_A == False:
            G=GraphONV(A) # graph orienté non valué
        class Table:  # class tableau
            def __init__(self,root):  # initialisation
                for i in range(total_rows):
                    for j in range(total_columns):
                        # ajouter la sortie du methode Pdic dans la table qui va pop up
                        self.e = Entry(root, width=20, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst = dictodic(G.Pdic())   # lst est la diste qui contien le tableau de dictionnaire
        total_rows = len(lst)       # le nombre des ligne
        total_columns = len(lst[0])  # le nombre des colones
        # afficher la fenetre qui contien le tableu de dictionnaire des des predecesseurs
        root = Tk()
        root.title("Dictionnaire des predecesseurs")  # titre
        t = Table(root)
        root.mainloop()# afficher le tableau

    def gaph_degres_fun(self):
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)   # graph Orienté , Valué
            self.graph_E ["text"] ="La notion du graph eulerien n'existe pas dans les graphs orienté"
        elif S_A == False and V_A == False:
            G=GraphONV(A)    # graph Orienté ,None Valué
            self.graph_E ["text"] ="La notion du graph eulerien n'existe pas dans les graphs orienté"
        elif S_A == True and V_A == True:
            G=GraphNOV(A)   # graph Non Orienté , Valué
        elif S_A == True and V_A == False:
            G=GraphNONV(A)    # graph Non Orienté ,None Valué
        class Table:  # class tableau
            def __init__(self,root):  # initialisation
                for i in range(total_rows):
                    for j in range(total_columns):
                        # ajouter la sortie du methode Pdic dans la table qui va pop up
                        self.e = Entry(root, width=20, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst,D=[],[]
        for i in range(len(G.matrix)):
            lst.append((i,G.degre(i)))
            D.append(G.degre(i))     # ajouter les degrée de chaque sommet à la liste D
        if self.graph_E ["text"] !="La notion du graph eulerien n'existe pas dans les graphs orienté" and all(d%2==0 for d in D):
            # si le graph est non orienté et tous les restes de division des degrée de leurs sommets = 0 donc il est eulerien
            self.graph_E ["text"] ="Eulerien"    
        elif self.graph_E ["text"] !="La notion du graph eulerien n'existe pas dans les graphs orienté" and type(G.SE())==list:
            # si le graph est non orienté et tous les restes de division des degrée de leurs sommets = 0  sauf exactement 2 sommet donc il est semi eulerien
            print("Semi eulerien avec début,fin="+"  "+str(G.SE()[0]-1)+" , "+str(G.SE()[1]-1))
            self.graph_E ["text"] ="Semi eulerien avec début,fin="+"  "+str(G.SE()[0]-1)+" , "+str(G.SE()[1]-1)  # afficher les extrimité
        elif self.graph_E ["text"] !="La notion du graph eulerien n'existe pas dans les graphs orienté":
            # si le graph est non orienté et  ni eulerien ni Semi eulerien
            self.graph_E ["text"] ="Ni eulerien ni Semi eulerien"   
        total_rows = len(lst)       # le nombre des ligne
        total_columns = len(lst[0])  # le nombre des colones
        # afficher la fenetre qui contien le tableu de dictionnaire des des predecesseurs
        root = Tk()
        root.title("Degres") # titre
        t = Table(root)
        root.mainloop()   # afficher le tableau

    def gaph_dijkstra_fun(self):   # commande du boutton dijkstra
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        #adapter le text avec l'algorithm de dijkstra
        abc=[] 
        for e,i in zip(A,range(len(A))):
            abc.append([])
            for f in e:
                if f==float("inf"): 
                    abc[i].append("inf")   # ajouter infinie
                elif f>=0:
                    abc[i].append(f)   # ajouter le nombre
                elif f<0:
                    #s'il ya un nombre n'egatif l'alerter et quiter 
                    self.graph_alert ["text"] =" Dijkstra ne marche pas avec les valeurs negatifs"
                    self.graph_alert ["bg"] ="RED"
                    return None
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(abc)   # graph Orienté , Valué
        elif S_A == False and V_A == False:
            G=GraphONV(abc)
            # graph Orienté ,Non Valué , dijkstra ne marche pas
            self.graph_alert ["text"] ="Dans cette version Dijkstra marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        elif S_A == True and V_A == True:
            G=GraphNOV(abc)    # graph Non Orienté , Valué
        elif S_A == True and V_A == False:
            G=GraphNONV(abc)
            # graph Non orienté Orienté ,Non Valué , dijkstra ne marche pas
            self.graph_alert ["text"] ="Dans cette version Dijkstra marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        class Table:  # class tableau
            def __init__(self,root):  # initialisation
                for i in range(total_rows):
                    for j in range(total_columns):
                        # ajouter la sortie du methode Pdic dans la table qui va pop up
                        self.e = Entry(root, width=35, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=j, column=i)
                        self.e.insert(END, lst[i][j])
        try:    # essayer d'executé la fonction di dijkstra
            lst = G.dijkstra(int(self.graph_S.get(1.0,END)))
        except ValueError:     # s'il y'a un erreur de valeur alerter qu'il faut entrer un sommet de debut
            self.graph_alert ["text"] ="Ecrir un Somet de début"
            self.graph_alert ["bg"] ="ORANGE"
        lst = G.dijkstra(int(self.graph_S.get(1.0,END)))    # affecter a la liste qui va s'afficher le tableau de dijkstra
        total_rows = len(lst)    # nombre des lignes
        total_columns = len(lst[0])   # nombre des colones
        root = Tk()
        root.title("Dijkstra")  # titre
        t = Table(root)
        root.mainloop()  # afficher le tableau

    def gaph_BF_fun(self):   # commande de boutton bellmanford et arboressance pour plus court et plus long chemain
        plt.figure(self.fig)
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        if self.plchvar.get()==1:  # si on a selectionner d'utiluser le lus long chemain
            for i in range(len(A)):
                for j in range(len(A)):
                    if A[i][j]=="inf" or A[i][j]==float("inf"):
                        pass
                    else:
                        A[i][j]=-A[i][j]    # inverser le signe de tous les nombres finie exsiste dans la matrice
        
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)   # graph Orienté , Valué
        elif S_A == False and V_A == False:
            # graph Orienté ,None Valué , alerter que bellman ford ne marche pas avec ce type de graphs et arreter
            G=GraphONV(A)
            self.graph_alert ["text"] ="Dans cette version Bellman ford marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        elif S_A == True and V_A == True:
            G=GraphNOV(A)   # graph Non Orienté , Valué
        elif S_A == True and V_A == False:
            # graph Non Orienté ,Non Valué , alerter que bellman ford ne marche pas avec ce type de graphs et arreter
            G=GraphNONV(A)
            self.graph_alert ["text"] ="Dans cette version Bellman ford marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        class Table:  # class Table
            def __init__(self,root): # initialisation
                for j in range(total_columns):
                    for i in range(total_rows):
                        self.e = Entry(root, width=35, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        if j == 1: # pour la premiere colone adapter le systeme d'indexation de 1.. à 0..
                            k = []
                            for h in range(len(lst[i][j])):
                                if lst[i][j][h] == None:
                                    k.append(None)       # si le predecesseurs est None pass
                                else:
                                    k.append(lst[i][j][h] -1)  # si non soustracte 1
                            self.e.insert(END, k)
                        else:    # pour les autres collones  ajoouter dans la tables 
                            self.e.insert(END, lst[i][j])
        # afficher L'arboressance
        try:
            Matrice= G.matrix    # initialiser Matrix comme la matrice adjacente
            lstm = G.BellmanFord(int(self.graph_S.get(1.0,END))+1)   # executé l'algorithm de bellman ford
            ARBR=[]  # initialiser ARBR
            # algorithm de passage de la derniere ligne d'algorithme de bellman ford vers l'arbdoressance
            # pour dst dans les distance et pred dans les predecesseurs et i comme index
            for dst,Pred,i in zip(lstm[0][-1][-1],lstm[1][-1][-1],range(len(lstm[0][-1][-1]))):  
                ARBR.append([])
                for j in range (len(lstm[0][-1][-1])):
                    try:  #essayer
                        if j == int(Pred)-1:
                            ARBR[i].append(Matrice[j][i])  # si j = predecesseur ajouter au ARBR la distance entre eux
                        else:
                            ARBR[i].append(float('inf'))   # si non ajouter infinie
                    except:
                        ARBR[i].append(float('inf'))   # s'il y'a un erreur ajouter linfinie
            # ARBR est maintenant la matrice adjacente de l'arboressance
            ARBR=np.array(ARBR).T   # transformer ARBR en module numpy et le transposé
            g=GraphOV(ARBR)   # g est l'arboressance -> graph orienté valué
            n = g.Ordre()   # n est l'ordre de ce graph
            a = g.matrix   # a est sa matrice 
            g = nx.DiGraph()  # g maintenant est Directer Graph depuis la biblioteque networkx
            edge_labels={}    # initialiser le dictionnaire des distances entre les sommets
            for i in range(n):
                for j in range(n):
                    if a[i][j] <float("inf"):
                        if i==j and A[i][i]==0:   
                            pass    # si nous somme dans la diagonal pass et A[i][i]==0 ne faire aucune chose
                        elif i==j and a[i][j]!=0:  # si nous somme dans la diagonal pass et a[i][i]!=0  
                            g.add_edge(i,j)   # ajouter les sommet
                            if self.logvar.get()==1 and self.plchvar.get()==1:
                                edge_labels[(i,j)]=str(round(math.exp(-a[i][j]),2)) # si on aplique le produit le le plus long chemin
                            elif self.logvar.get()==0 and self.plchvar.get()==1:
                                edge_labels[(i,j)]=str(round(-a[i][j],2))  # si on applique le plus long chemin
                            elif self.logvar.get()==0 and self.plchvar.get()==0:
                                edge_labels[(i,j)]=str(round(a[i][j],2))    # si on n'applique aucun des deux 
                            else:
                                edge_labels[(i,j)]=str(round(math.exp(a[i][j]),2)) # si on applique seulement le log
                        elif not((j,i) in edge_labels.keys()): # si on a deja ajouter la distance ne rajouter pas 
                            g.add_edge(i,j)
                            if self.logvar.get()==1 and self.plchvar.get()==1:
                                edge_labels[(i,j)]=str(round(math.exp(-a[i][j]),2)) # si on aplique le produit le le plus long chemin
                            elif self.logvar.get()==0 and self.plchvar.get()==1:
                                edge_labels[(i,j)]=str(round(-a[i][j],2))  # si on applique le plus long chemin
                            elif self.logvar.get()==0 and self.plchvar.get()==0:
                                edge_labels[(i,j)]=str(round(a[i][j],2))    # si on n'applique aucun des deux 
                            else:
                                edge_labels[(i,j)]=str(round(math.exp(a[i][j]),2)) # si on applique seulement le log
            pos = nx.spring_layout(g)  # posiotions des sommets
            # dessiner les sommet et les arcs
            nx.draw(g, pos, edge_color='black', width=1, linewidths=1,node_size=500, node_color='pink', alpha=0.9,labels={node: node for node in g.nodes()})    
            nx.draw_networkx_edge_labels(g, pos, edge_labels = edge_labels,font_color='red')
        except RecursionError: # s'il y'a un erreur de recursivité donc il y'a un circuit absorbant
            self.graph_alert ["text"] ="Il'ya un circuit abssorbant"
            self.graph_alert ["bg"] ="RED"
        except ValueError:    # s'il y'a un erreur de valeur donc il faut entrer le sommet de debut
            self.graph_alert ["text"] ="Ecrir un Somet de début"
            self.graph_alert ["bg"] ="ORANGE"
        lst=[]
        #ajouter à lst le tableaux de bellman ford selon les deux cas: plus long ou plus cout chemain
        for f,g in zip(lstm[0],lstm[1]):
            for h,k in zip(f,g):
                if self.plchvar.get()==1:
                    var=[]
                    for i in range(len(h)):
                        try:    # essayer d'inverser le signe si on a dans le plus long chemain
                            if self.logvar.get()==1:
                                #adaptation avec les nombres entiers si la longueur est finie
                                if math.exp(-h[i])-int(math.exp(-h[i]))>0.5:
                                    var.append(int(math.exp(-h[i])+1))# majorer la distance
                                else:
                                    var.append(int(math.exp(-h[i]))) # minorer la distance
                            else:
                                var.append(-h[i])  # si on a la somme on inverse le signe seulement 
                        except:
                            var.append(h[i])   # s'il y'a un erreur n'inverse pas le signe
                    lst.append((var,k))   # ajouter le tuple (tableaux L, tableaux P) à lst
                else:
                    lst.append((h,k))   # ajouter le tuple (tableaux L, tableaux P) à lst
                
        total_rows = len(lst)    # nombre des lignes
        total_columns = len(lst[0])    # nombre des colones
        root = Tk()
        if self.plchvar.get()==1:
            root.title("Bellman Ford plus long chemain")    # titre plch
        else:
            root.title("Bellman Ford plus court chemain")    # titre pcchm
        t = Table(root)
        self.fig = self.fig +1   # incrementer la figure
        plt.show()     # afficher l'arboraissance
        root.mainloop()  # afficher le tableau

    def gaph_plot_fun(self):  #
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        plt.figure(self.fig)
        # a est le texte ecrit
        a = self.graph_text.get(1.0,END)
        # A est la matrice ou l'erreur retourner de ttm 
        A=ttm(a,self.logvar.get())
        if type(A)==str:   # si A est un texte l'afficher dans l'alert
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        # detection du type de graph   S_A detection s'il est symetrique    S_V detection s'il est valué
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)   # orienté valué    """""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            n = G.Ordre()                     #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            A = G.matrix                       #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            G = nx.DiGraph()   # ici directed graph
            edge_labels={}
            for i in range(n):
                for j in range(n):
                    if A[i][j] <float("inf"): 
                        if i==j and A[i][i]==0:
                            pass
                        elif i==j and A[i][j]!=0:
                            G.add_edge(i,j)
                            edge_labels[(i,j)]=str(A[i][j])
                        elif not((j,i) in edge_labels.keys()):
                            G.add_edge(i,j)
                            edge_labels[(i,j)]=str(A[i][j])
            pos = nx.spring_layout(G)
            nx.draw(G, pos, edge_color='black', width=1, linewidths=1,node_size=500, node_color='pink', alpha=0.9,labels={node: node for node in G.nodes()})            
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,font_color='red')
            self.fig = self.fig +1
            plt.show()


            
            return None
        elif S_A == False and V_A == False:
            G=GraphONV(A)   # orienté Non valué    """""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            n = G.Ordre()                     #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            A = G.matrix                       #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            G = nx.DiGraph(A)   # ici directed graph
            for i in range(n):
                for j in range(n):
                    if A[i][j] == 1: 
                        G.add_edge(i,j)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, edge_color='black', width=1, linewidths=1,node_size=500, node_color='pink', alpha=0.9,labels={node: node for node in G.nodes()})
            self.fig = self.fig +1
            plt.show()
            
        elif S_A == True and V_A == True:
            G=GraphNOV(A)   # Non orienté valué    """""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            n = G.Ordre()                     #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            A = G.matrix                       #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            G = nx.Graph()    # ici graph
            edge_labels={}
            for i in range(n):
                for j in range(n):
                    if A[i][j] <float("inf"):
                        print("added") 
                        if i==j and A[i][i]==0:
                            pass
                        elif i==j and A[i][j]!=0:
                            G.add_edge(i,j)
                            edge_labels[(i,j)]=str(A[i][j])
                        elif not((j,i) in edge_labels.keys()):
                            G.add_edge(i,j)
                            edge_labels[(i,j)]=str(A[i][j])
            pos = nx.spring_layout(G)
            nx.draw(G, pos, edge_color='black', width=1, linewidths=1,node_size=500, node_color='pink', alpha=0.9,labels={node: node for node in G.nodes()})            
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,font_color='red')
            self.fig = self.fig +1
            plt.show()


            
        elif S_A == True and V_A == False:
            G=GraphNONV(A)   # Non orienté Non valué    """""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            n = G.Ordre()                     #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            A = G.matrix                       #"""""""""""" meme commentaire que pour l'arboraissance dans bellman ford commande boutton
            G = nx.Graph(A)    # ici graph
            for i in range(n):
                for j in range(n):
                    if A[i][j] == 1:
                        G.add_edge(i,j)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, edge_color='black', width=1, linewidths=1,node_size=500, node_color='pink', alpha=0.9,labels={node: node for node in G.nodes()})
            self.fig = self.fig +1
            plt.show()



    # presque meme commentaire que por les commande si dessous
    def gaph_FW_fun(self):  # commande de boutton Floyd warshall
        # initialiser le label alert
        self.graph_alert ["text"] =" Alert"
        self.graph_alert ["bg"] ="WHITE"
        a = self.graph_text.get(1.0,END)
        A=ttm(a,self.logvar.get())
        if type(A)==str:
            self.graph_alert ["text"] = A
            self.graph_alert ["bg"] ="RED"
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)
        elif S_A == False and V_A == False:
            G=GraphONV(A)
            self.graph_alert ["text"] ="Dans cette version Floyd Warshall marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        elif S_A == True and V_A == True:
            G=GraphNOV(A)
        elif S_A == True and V_A == False:
            G=GraphNONV(A)
            self.graph_alert ["text"] ="Dans cette version Floyd Warshall marche seulemenet avec les graphs valué"
            self.graph_alert ["bg"] ="YELLOW"
            return None
        class Table:
            def __init__(self,root):
                for i in range(total_columns):
                    for j in range(total_rows):
                        self.e = Entry(root, width=8, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst= G.FloydWarshall()
        total_rows = len(lst)
        total_columns = len(lst[0])
        root = Tk()
        root.title("Floyd Warshall")
        t = Table(root)
        root.mainloop()
            
    """def tool(self):
        self.graph_alert ["text"] =" alert"
        self.graph_alert ["bg"] ="WHITE"
        a = self.graph_text.get(1.0,END)
        A=ttm(a,self.logvar.get())
        S_A , V_A = S_detect(A) , V_detect(A)
        if S_A == False and V_A == True:
            G=GraphOV(A)
        elif S_A == False and V_A == False:
            G=GraphONV(A)
        elif S_A == True and V_A == True:
            G=GraphNOV(A)
        elif S_A == True and V_A == False:
            G=GraphNONV(A)
        class Table:
            def __init__(self,root):
                # code for creating table
                for i in range(total_columns):
                    for j in range(total_rows):
                        self.e = Entry(root, width=20, fg='blue',font=('Arial',16,'bold'))
                        self.e.grid(row=i, column=j)
                        self.e.insert(END, lst[i][j])
        lst = dictodic(G.Pdic())
        total_rows = len(lst)
        total_columns = len(lst[0])
        root = Tk()
        root.title("k")
        t = Table(root)
        root.mainloop()









# test


test1=[[0,1,1,0,0],[1,1,1,1,1],[1,1,1,0,1],[0,1,0,0,1],[0,1,1,1,0]] # NONV
test2=[[0,1,1,0,0],[0,1,0,1,0],[0,1,1,0,0],[0,1,0,0,1],[0,1,1,0,0]] # ONV
test3=[[0,1,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,1,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[1,0,1,0,0,0]] # ONV
test4=[[0,19,7,"inf","inf"],[19,0,5,14,22],[7.0,5,0,12,"inf"],["inf",14,12,0,"inf"],["inf",22,"inf","inf",0]] #NOV
test5=[[6,"inf",7],[5,"inf",1],["inf",3,"inf"]] # OV
test6=[[0,1,0,1,0],[1,0,0,0,1],[0,0,0,1,1],[1,0,1,0,0],[0,1,1,0,0]] #NONV E H
test7=[[0,1,0,0,0,0],[1,0,1,1,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,1,1,0,1],[0,0,0,0,1,0]] #NONV SE
test8=[[0,1,0,1,1],[1,0,1,0,1],[0,1,0,1,1],[1,0,1,0,1],[1,1,1,1,0]] #NONV H
test9=[[0,7,1,"inf","inf","inf"],["inf",0,"inf",4,"inf",1],["inf",5,0,"inf",2,7],["inf","inf","inf",0,"inf","inf"],["inf",2,"inf",5,0,"inf"],["inf","inf","inf","inf",3,0]]#OV dijkstra
test10=[[0,2,"inf",4,"inf","inf","inf","inf"],["inf",0,"inf",-1,4,5,"inf","inf"],[-3,"inf",0,"inf","inf","inf",1,"inf"],["inf","inf",2,0,"inf","inf","inf","inf"],["inf","inf","inf","inf",0,"inf","inf",2],["inf","inf","inf","inf","inf",0,3,"inf"],["inf","inf","inf",-2,"inf","inf",0,2],["inf","inf","inf","inf","inf",-3,"inf",0]] # OV BellmanFord
test11=[[0,12,2,1,1],[1,0,1,0,1],[0,1,0,1,1],[1,0,1,0,1],[1,1,1,1,0]]
test12=[[6,float("inf"),7],[5,float("inf"),1],[float("inf"),3,float("inf")]]

testG=GraphOV(test9)

#print(testG.Taille())


"""
A=[[0,5,3,2,float("inf"),float("inf"),float("inf"),float("inf")],
    [5,0,1,float("inf"),2,2,float("inf"),float("inf")],
    [3,1,0,1,float("inf"),float("inf"),float("inf"),float("inf")],
    [2,float("inf"),1,0,3,float("inf"),2,float("inf")],
    [float("inf"),2,float("inf"),3,0,4,4,float("inf")],
    [float("inf"),2,float("inf"),float("inf"),4,0,1,7],
    [float("inf"),float("inf"),float("inf"),2,4,1,0,6],
   [float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),7,6,0]]

L=[0, 4.0, 3.0, 2.0, 5.0, 5.0, 4.0, 10.0]
P=[1, 3, 1, 1, 4, 7, 4, 7]

# algorithm de passage de la derniere ligne d'algorithme de dijksta vers l'arbdoressance
def dijkstra_arboressance(L,P,Matrice):
    # L et P sont la derniere ligne de tableaux de dijkstra et Matrice c'est la matrice agjacente du graph
    ARBR=[]  # initialiser ARBR
    # pour dst dans les distance et pred dans les predecesseurs et i comme index
    for dst,Pred,i in zip(L,P,range(len(L))):  
        ARBR.append([])
        for j in range (len(L)):
            try:  #essayer
                if j == int(Pred)-1:
                    ARBR[i].append(Matrice[j][i])  # si j = predecesseur ajouter au ARBR la distance entre eux
                else:
                    ARBR[i].append(float('inf'))   # si non ajouter infinie
            except:
                ARBR[i].append(float('inf'))   # s'il y'a un erreur ajouter linfinie
    # ARBR est maintenant la matrice adjacente de l'arboressance
    return np.array(ARBR).T

#print(np.array(dijkstra_arboressance(L,P,A)))
# run


if __name__ == "__main__":
    man = Main()
    man.mainloop()








































