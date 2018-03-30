import string
import math
import nltk
import pickle as cPickle

"""
This file contains the functions to truecase a sentence.
"""

class TrueCaser:
    def __init__(self, obj_path):
        with open(obj_path, 'rb') as f:
            self.uniDist = cPickle.load(f)
            self.backwardBiDist = cPickle.load(f)
            self.forwardBiDist = cPickle.load(f)
            self.trigramDist = cPickle.load(f)
            self.wordCasingLookup = cPickle.load(f)

        self.outOfVocabularyTokenOption = "title"

    def getScore(self, prevToken, possibleToken, nextToken):
        pseudoCount = 5.0
        
        #Get Unigram Score
        nominator = self.uniDist[possibleToken]+pseudoCount    
        denominator = 0    
        for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
            denominator += self.uniDist[alternativeToken]+pseudoCount
            
        unigramScore = nominator / denominator
            
            
        #Get Backward Score  
        bigramBackwardScore = 1
        if prevToken != None:  
            nominator = self.backwardBiDist[prevToken+'_'+possibleToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.backwardBiDist[prevToken+'_'+alternativeToken]+pseudoCount
                
            bigramBackwardScore = nominator / denominator
            
        #Get Forward Score  
        bigramForwardScore = 1
        if nextToken != None:  
            nextToken = nextToken.lower() #Ensure it is lower case
            nominator = self.forwardBiDist[possibleToken+"_"+nextToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.forwardBiDist[alternativeToken+"_"+nextToken]+pseudoCount
                
            bigramForwardScore = nominator / denominator
            
            
        #Get Trigram Score  
        trigramScore = 1
        if prevToken != None and nextToken != None:  
            nextToken = nextToken.lower() #Ensure it is lower case
            nominator = self.trigramDist[prevToken+"_"+possibleToken+"_"+nextToken]+pseudoCount
            denominator = 0    
            for alternativeToken in self.wordCasingLookup[possibleToken.lower()]:
                denominator += self.trigramDist[prevToken+"_"+alternativeToken+"_"+nextToken]+pseudoCount
                
            trigramScore = nominator / denominator
            
        result = math.log(unigramScore) + math.log(bigramBackwardScore) + math.log(bigramForwardScore) + math.log(trigramScore)
        #print "Scores: %f %f %f %f = %f" % (unigramScore, bigramBackwardScore, bigramForwardScore, trigramScore, math.exp(result))
    
    
        return result

    def getTrueCase(self, tokens):
        """
        Returns the true case for the passed tokens.
        @param tokens: Tokens in a single sentence
        @param outOfVocabulariyTokenOption:
            title: Returns out of vocabulary (OOV) tokens in 'title' format
            lower: Returns OOV tokens in lower case
            as-is: Returns OOV tokens as is
        """
        tokensTrueCase = []
        for tokenIdx in list(range(len(tokens))):
            token = tokens[tokenIdx]
            if token in string.punctuation or token.isdigit():
                tokensTrueCase.append(token)
            else:
                if token in self.wordCasingLookup:
                    if len(self.wordCasingLookup[token]) == 1:
                        tokensTrueCase.append(list(self.wordCasingLookup[token])[0])
                    else:
                        prevToken = tokensTrueCase[tokenIdx-1] if tokenIdx > 0  else None
                        nextToken = tokens[tokenIdx+1] if tokenIdx < len(tokens)-1 else None
                        
                        bestToken = None
                        highestScore = float("-inf")
                        
                        for possibleToken in self.wordCasingLookup[token]:
                            score = self.getScore(prevToken, possibleToken, nextToken)
                            if score > highestScore:
                                bestToken = possibleToken
                                highestScore = score
                            
                        tokensTrueCase.append(bestToken)
                        
                    if tokenIdx == 0:
                        tokensTrueCase[0] = tokensTrueCase[0].title();
                        
                else: #Token out of vocabulary
                    if self.outOfVocabularyTokenOption == 'title':
                        tokensTrueCase.append(token.title())
                    elif self.outOfVocabularyTokenOption == 'lower':
                        tokensTrueCase.append(token.lower())
                    else:
                        tokensTrueCase.append(token) 
        
        return tokensTrueCase

# tc = TrueCaser("distributions.obj")
# question = "where is china?"
# tokens = nltk.word_tokenize(question)
# tokens = [token.lower() for token in tokens]
# question = " ".join(tc.getTrueCase(tokens))
# print(question)


