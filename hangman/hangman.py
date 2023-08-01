import random
import os

clear = lambda: os.system('cls')

pics=[' ',
    '''
=========||
    |    ||
         ||
         ||
         ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
         ||
         ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
    |    ||
         ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
   /|    ||
         ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
   /|\   ||
         ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
   /|\   ||
   /     ||
         ||
         || ''',
'''
=========||
    |    ||
    0    ||
   /|\   ||
   / \   ||
         ||
         || ''',
]

with open("words.txt","r") as f1:
    word=f1.read()
    word=word.lower()
    word=word.split()

def getWord(WordList):
    Index=random.randint(0,len(WordList)-1)
    return WordList[Index]
def display(Correct,Missed,SecretWord):
    clear()
    print (pics[len(Missed)])
    print("Lista de letras utilizadas ")
    for Letter in Missed+Correct:
        print (Letter,end=' ')
    Blanks='_'*len(SecretWord)
    for i in range(len(SecretWord)):
        if SecretWord[i] in Correct:
            Blanks=Blanks[:i]+SecretWord[i]+Blanks[i+1:]
    print ("\n\n")
    print("La palabra tiene ",end='')
    print(len(SecretWord),end=' ')
    print("caracteres de longitud")
    for i in range (0,len(Blanks)):
        print (Blanks[i],end=' ')
    print()
    ch=''
    if Blanks == SecretWord:
        print("¡Ganaste!")
        print("¿Jugar otra vez? y/n ")
        ch=input()
        guess(ch)
    elif Blanks !=SecretWord and len(Missed)==7:
        print("Perdiste\n")
        print ("La palabra era ",end='')
        print(SecretWord)        
        print("¿Jugar otra vez? y/n ")
        ch=input()
        guess(ch)        
def guess(Choice):
    if Choice=='y':
        print('¡Bienvenido!\n')
        Missed=''
        Correct=''
        SecretWord=getWord(word)
        print('_ '*len(SecretWord))
        while len(Missed)<len(pics)-1 :
            print('Por favor ingrese una letra...\n')
            letter=input()
            flag=checkLetter(letter,Missed,Correct)
            if flag == 1 and letter in SecretWord:
                Correct=Correct+letter
                display(Correct,Missed,SecretWord)
            elif flag==1 and letter not in SecretWord:
                Missed=Missed+letter
                display(Correct,Missed,SecretWord)
    else:
        exit("\nGracias por jugar")
def checkLetter(letter,Missed,Correct):
    UnForbidden='abcdefghijklmnopqrstuvwxyz'
    if letter in UnForbidden and len(letter) == 1 and letter not in Missed+Correct:
        return 1
    elif letter not in UnForbidden:
        print("Por favor solo ingrese letras\n")
        return 2
    elif len(letter) !=1:
        print("Solo se permite ingresar una letra\n")
        return 3
    elif letter in Missed+Correct:
        print("Ya utilizó esa letra\n")
        return 4
guess('y')