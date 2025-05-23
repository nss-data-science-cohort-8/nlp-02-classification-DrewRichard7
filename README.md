# Classification of Consumer Complaints

The Consumer Financial Protection Bureau publishes the Consumer Complaint Database, a collection of complaints about consumer financial products and services that were sent to companies for response. Complaints are published after the company responds, confirming a commercial relationship with the consumer, or after 15 days, whichever comes first.

You have been provided with a dataset of over 350,000 such complaints for 5 common issue types. Your goal is to train a text classification model to identify the issue type based on the consumer complaint narrative. The data can be downloaded from <https://drive.google.com/file/d/1Hz1gnCCr-SDGjnKgcPbg7Nd3NztOLdxw/view?usp=share_link>

As you work, answer the following questions:

* What steps did you take to preprocess the data?
* How did a model using unigrams compare to one using bigrams or trigrams?
* How did a count vectorizer compare to a tfidf vectorizer?
* What models did you try and how successful were they? Where did they struggle? Were there issues that the models commonly mixed up?
* What words or phrases were most influential on your models' predictions?

**Bonus:** A larger dataset containing 20 additional categories can be downloaded from <https://drive.google.com/file/d/1gW6LScUL-Z7mH6gUZn-1aNzm4p4CvtpL/view?usp=share_link>. How well do your models work with these additional categories?
