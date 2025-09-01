#number guessing
# lets the user choose a random number between 1 and 10(both included) and guesses it.
# suggestion: try to take the upper and lower limit from the user and change the code to guess it!


import random
print("Choose a number between 1 and 10, and let me guess it")
k=True
while(k==True):
  print(random.randint(1,10))
  x=int(input("Press 1 if the guess is right, else press any other number"))
  if(x==1):
    print("I guessed it right!")
    k=False