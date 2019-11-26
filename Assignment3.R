install.packages('ISLR')
install.packages('randomForest')
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("caret") 

install.packages("psych")

library(psych)
library(caret) 
# Obtain the dataset "College" from the "ISLR" packages
library(ISLR)
library(ggplot2)
library(gridExtra)


df <- read.csv("gastext.csv",na.strings=c("NA",""),stringsAsFactors = F)

summary(df)
str(df)
multi.hist(df[,3:15])

# Highly skewed:
#   
# Comp_card_flag
# AcctType_flag
# HQ_flag
# Multi_flag
df[,3:15]<-lapply(df[,3:15],factor)
df = df[, !(colnames(df) %in% c("Comp_card_flag", "AcctType_flag", "HQ_flag", "Multi_flag"))]

# p0  = ggplot(df, aes(x=df$Cust_ID           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
# p1  = ggplot(df, aes(x=df$Comment           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
# p2  = ggplot(df, aes(x=df$Target           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p3  = ggplot(df, aes(x=df$Service_flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p4  = ggplot(df, aes(x=df$CustType_flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p5  = ggplot(df, aes(x=df$Contact_flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p6  = ggplot(df, aes(x=df$new_flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p7  = ggplot(df, aes(x=df$Choice_flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p8  = ggplot(df, aes(x=df$Loyal_Status           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p9  = ggplot(df, aes(x=df$Contact_Flag2           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")
p10  = ggplot(df, aes(x=df$NewCust_Flag           , color=Target))  + geom_histogram(binwidth=1, stat="count", fill="white")


grid.arrange(
  #p0, p1 , p2 ,
  p3 , p4 
  , p5 , p6 , p7 
  , p8 , p9 , p10
  , ncol = 4)

# Data pre-processing
install.packages('quanteda')
library(quanteda)
myCorpus <- corpus(df$Comment)
summary(myCorpus)

# Create a dfm
myDfm <- dfm(myCorpus)
topfeatures(myDfm)


# Create an alternative dfm based on bigram
myTokens <- tokens(myCorpus)
bigram <- tokens_ngrams(myTokens,n=1:2)
myDfm_bigram <- dfm(bigram)
View(myDfm_bigram)

# Remove stop words
#install.packages("stopwords")
library(stopwords)
myDfm <- dfm(myCorpus, remove = c(stopwords("english")))
topfeatures(myDfm)
###########################
#dfmat1 <- dfm(data_corpus_irishbudget2010)

# lemmatization
lis <- c("productx")
lemma <- rep("product", length(lis))
myDfm <- dfm_replace(myDfm, pattern = lis, replacement = lemma)
featnames(dfm_select(myDfm, pattern = lis))

# stemming
# feat <- featnames(myDfm)
# featstem <- char_wordstem(feat, "porter")
# dfmat3 <- dfm_replace(myDfm, pattern = feat, replacement = featstem, case_insensitive = FALSE)
# identical(dfmat3, dfm_wordstem(dfmat1, "porter"))

###################################


# Remove an additional user-defined stop word and Perform stemming
myDfm <- dfm(myCorpus, 
             remove = c(stopwords("english"),'.',',','t',"�","use","get","can","1","2","3","4","/","?","1000","\"","!","s","(",")","-","done")
             ,stem = T)
             
topfeatures(myDfm)

dim(myDfm)
View(myDfm)

# Weight a dfm by tf-idf
myDfm_tfidf <- dfm_tfidf(myDfm)
View(myDfm_tfidf)

# Perform SVD
mySvd <- textmodel_lsa(myDfm_tfidf)
mySvd$docs[]

# Predict SVD for a new sentence
newDfm <- dfm(c('My puppy is very cute'))
#newDfm

newDfm <- dfm_select(newDfm, pattern = myDfm)
newDfm

# Technically we should use newDfm_tfidf
# However, tf_idf becomes NA for a single document corpus
# newDfm_tfidf <- dfm_tfidf(newDfm)
newSvd <- predict(mySvd,newdata=newDfm)
newSvd$docs[]

# Topic modeling
install.packages('topicmodels')
install.packages('tidytext')
library(topicmodels)
library(tidytext)

# Topic modeling based on the original DFM
myLda <- LDA(myDfm,k=20,control=list(seed=101))
myLda

# Term-topic probabilities
myLda_td <- tidy(myLda)
myLda_td

# Visulize most common terms in each topic
library(ggplot2)
library(dplyr)

top_terms <- myLda_td %>%
  group_by(topic) %>%
  top_n(7, beta) %>%
  ungroup() %>%
#  arrange(topic, -beta)
  arrange(-beta, topic)

top_terms

top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# log_ratio calculated differences between two topics
library(tidyr)

myLda_td

beta_spread <- myLda_td %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

topic1

head(beta_spread)

# Visulize the greatest differences between the two topics
beta_spread %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip()

textplot_wordcloud(myDfm,max_words=200)

# Document-topic probabilities
ap_documents <- tidy(myLda, matrix = "gamma")
ap_documents

# Document similarity for "text1"
text_sim <- textstat_simil(myDfm, 
                           selection="text1",
                           margin="document",
                           method="correlation")

head(as.matrix(text_sim),5)

# Perform hierachical clustering for documents
doc_dist <- textstat_dist(myDfm)
clust <- hclust(as.dist(doc_dist))
plot(clust,xlab="Distance",ylab=NULL)

# Term similarity
term_sim <- textstat_simil(myDfm,
                           selection=c("servic"),
                           margin="feature",
                           method="correlation")
head(as.matrix(term_sim),5)

as.matrix(term_sim)

term_sim


myDfm <- dfm_remove(myDfm, c('shower','point','don','productx'))
myDfm <- as.matrix(myDfm)
myDfm <-myDfm[which(rowSums(myDfm)>0),]
myDfm <- as.dfm(myDfm)


myLda <- LDA(myDfm,k=4,control=list(seed=101))
myLda


top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# Visulize the greatest differences between the two topics
beta_spread %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip()

######################################################
#  decision tree model without text mining
######################################################
install.packages('rpart')
install.packages('rpart.plot')
library(caTools)
library(ROCR)
library(rpart)
library(rpart.plot)



df <- read.csv("gastext.csv",na.strings=c("NA",""),stringsAsFactors = F)

summary(df)
str(df)
multi.hist(df[,3:15])

# Highly skewed:
#   
# Comp_card_flag
# AcctType_flag
# HQ_flag
# Multi_flag
df[,3:15]<-lapply(df[,3:15],factor)
df = df[, !(colnames(df) %in% c("Comp_card_flag", "AcctType_flag", "HQ_flag", "Multi_flag","Comment","Cust_ID"))]
df = df[, !(colnames(df) %in% c("Comment"))]

set.seed(101)

df$Target <- factor(df$Target)

sample <- sample.split(df$Target, SplitRatio = 0.7)

# Create Training Data
df.train = subset(df, sample == TRUE)

# Create Validation Data
df.valid = subset(df, sample == FALSE)

# Build a decision tree model
tree.model <- rpart(Target~.,method="class",data=df.train)

# Display decision tree results
printcp(tree.model)
# Display decision tree plot
prp(tree.model,type=2,extra=106)

#Evaluation model performance using the validation dataset
#Predict the default probabilities based on the validataion dataset
pred.probabilities <- predict(tree.model,df.valid)
pred.probabilities <- as.data.frame(pred.probabilities)

#Turn the default probabilities to binary, threshold is 0.5
joiner <- function(x){
  if (x<0.5){
    return('Yes')
  }else{
    return("No")
  }
}

pred.probabilities$Result <- sapply(pred.probabilities$`1`,joiner)

# Create the confusion matrix
print("The confusion matrix is:")
print(table(pred.probabilities$Result,df.valid$Target))

# Create the ROC curve
pred <- prediction(pred.probabilities$`1`,df.valid$Target)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(a=0,b=1)

# Calculate and print AUC value
auc <- performance(pred, measure="auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the baseline regression model is:", auc))


######################################################
#  decision tree model with text mining
######################################################

df <- read.csv("gastext.csv",na.strings=c("NA",""),stringsAsFactors = F)

summary(df)
str(df)

# Highly skewed:
#   
# Comp_card_flag
# AcctType_flag
# HQ_flag
# Multi_flag
df[,3:15]<-lapply(df[,3:15],factor)
df = df[, !(colnames(df) %in% c("Comp_card_flag", "AcctType_flag", "HQ_flag", "Multi_flag","Cust_ID"))]
##df = df[, !(colnames(df) %in% c("Comment"))]

set.seed(101)

df$Target <- factor(df$Target)


myCorpus <- corpus(df$Comment)
summary(myCorpus)

# Create a dfm
myDfm <- dfm(myCorpus)
topfeatures(myDfm)


# Create an alternative dfm based on bigram
myTokens <- tokens(myCorpus)
bigram <- tokens_ngrams(myTokens,n=1:2)
myDfm_bigram <- dfm(bigram)
#View(myDfm_bigram)

# Remove stop words
#install.packages("stopwords")
#library(stopwords)
myDfm <- dfm(myCorpus, remove = c(stopwords("english")))
topfeatures(myDfm)
###########################
#dfmat1 <- dfm(data_corpus_irishbudget2010)

# lemmatization
lis <- c("productx")
lemma <- rep("product", length(lis))
myDfm <- dfm_replace(myDfm, pattern = lis, replacement = lemma)
featnames(dfm_select(myDfm, pattern = lis))


# Remove an additional user-defined stop word and Perform stemming
myDfm <- dfm(myCorpus, 
             remove = c(stopwords("english"),'.',',','t',"�","use","get","can","1","2","3","4","/","?","1000","\"","!","s","(",")","-","done")
             ,stem = T)

topfeatures(myDfm)


#myDfm <- dfm_remove(myDfm, c('shower','point','don','productx'))
myDfm <- dfm_remove(myDfm, c('shower','point','don'))
myDfm <- as.matrix(myDfm)
#myDfm <-myDfm[which(rowSums(myDfm)>0),]
myDfm <- as.dfm(myDfm)

myDfm <- dfm_trim(myDfm,min_termfreq=4, min_docfreq = 2)

# Weight a dfm by tf-idf
myDfm_tfidf <- dfm_tfidf(myDfm)
#View(myDfm_tfidf)

# Perform SVD
mySvd <- textmodel_lsa(myDfm_tfidf, nd=10)
mySvd$docs[]

newDf = df[, !(colnames(df) %in% c( "Comment"))]
#n<-dim(newDf)[1]
#newDf<-newDf[1:(n-1),]
modelData <-cbind(newDf,as.data.frame(mySvd$docs))

#colnames(modelData)[1] <- "Author"
head(modelData)


# Technically we should use newDfm_tfidf
# However, tf_idf becomes NA for a single document corpus
# newDfm_tfidf <- dfm_tfidf(newDfm)
# newSvd <- predict(mySvd,newdata=newDfm)
# newSvd$docs[]
# 
# myLda <- LDA(myDfm,k=4,control=list(seed=101))
# myLda
# 
# # Term-topic probabilities
# myLda_td <- tidy(myLda)
# myLda_td
# 
# 
# top_terms <- myLda_td %>%
#   group_by(topic) %>%
#   top_n(7, beta) %>%
#   ungroup() %>%
#   #  arrange(topic, -beta)
#   arrange(-beta, topic)
# 
# top_terms
# 
# top_terms %>%
#   mutate(term = reorder(term, beta)) %>%
#   ggplot(aes(term, beta, fill = factor(topic))) +
#   geom_bar(stat = "identity", show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   coord_flip()
# 
# top_terms %>%
#   mutate(term = reorder(term, beta)) %>%
#   ggplot(aes(term, beta, fill = factor(topic))) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   coord_flip()



set.seed(1010)

df$Target <- factor(df$Target)

sample <- sample.split(df$Target, SplitRatio = 0.7)

# Create Training Data
df.train = subset(df, sample == TRUE)

# Create Validation Data
df.valid = subset(df, sample == FALSE)

df.train

# Build a decision tree model
##tree.model <- rpart(Target~.,method="class",data=df.train)

tree.model <- rpart(Target~., data=df.train, parms=list(split="information"), cp=.005)

rpart.plot(tree.model, box.palette="RdBu", shadow.col="gray", nn=TRUE)

# Display decision tree results
printcp(tree.model)
# Display decision tree plot
prp(tree.model,type=2,extra=106)

#Evaluation model performance using the validation dataset
#Predict the default probabilities based on the validataion dataset
pred.probabilities <- predict(tree.model,df.valid)
pred.probabilities <- as.data.frame(pred.probabilities)

#Turn the default probabilities to binary, threshold is 0.5
joiner <- function(x){
  if (x<0.5){
    return('Yes')
  }else{
    return("No")
  }
}

pred.probabilities$Result <- sapply(pred.probabilities$`1`,joiner)

# Create the confusion matrix
print("The confusion matrix is:")
print(table(pred.probabilities$Result,df.valid$Target))

# Create the ROC curve
pred <- prediction(pred.probabilities$`1`,df.valid$Target)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(a=0,b=1)

# Calculate and print AUC value
auc <- performance(pred, measure="auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the baseline regression model is:", auc))


