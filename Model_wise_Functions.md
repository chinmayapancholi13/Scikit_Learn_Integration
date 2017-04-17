## Integration with scikit-learn

This document has been created to serve as a blueprint for the development of wrappers for all the Gensim models while integrating Gensim with scikit-learn.

- **Author-Topic Model**

```python
class SklearnWrapperAuthorTopicModel(models.AuthorTopicModel, TransformerMixin, BaseEstimator):
    """
    Base Author Topic module
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, author2doc=None, doc2author=None,
            chunksize=2000, passes=1, iterations=50, decay=0.5, offset=1.0,
            alpha='symmetric', eta='symmetric', update_every=1, eval_every=10,
            gamma_threshold=0.001, serialized=False, serialization_path=None,
            minimum_probability=0.01, random_state=None):
        """
        Initialize the Author Topic model using the parameter values passed
        """
        # If corpus and one of author2doc/doc2author dictionaries are given,
        # we could start training right away. If not given, the model is left untrained
        # (probably because the user wants to call the `update` method manually later).

        pass


    def get_params(self, deep=True):
        """
        Return all the relevant parameters of the model as a dictionary
        """
        pass


    def set_params(self, **parameters):
        """
        Set all the parameters passed
        """
        for parameter, value in parameters.items():
            self.parameter = value
        return self


    def fit(self, X, y=None):
        """
        Fit the input data into the class object
        Call gensim.models.AuthorTopicModel
        """
        # set the corpus and doc2author/author2doc using the parameter passed

        # The model should have self.corpus and atleast one of self.doc2author/self.author2doc set

        models.AuthorTopicModel.__init__(self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
            author2doc=self.author2doc, doc2author=self.doc2author, chunksize=self.chunksize, passes=self.passes,
            iterations=self.iterations, decay=self.decay, offset=self.offset, alpha=self.alpha, eta=self.eta,
            update_every=self.update_every, eval_every=self.eval_every, gamma_threshold=self.gamma_threshold,
            serialized=self.serialized, serialization_path=self.serialization_path, minimum_probability=self.minimum_probability, random_state=self.random_state)

        return self

    def transform(self, data):
        """
        Return the topics corresponding to author(s) passed as the parameter
        """
        # we would be using the function `get_author_topics` similar to model.get_author_topics('author_name')

        pass


    def partial_fit(self, data):
        """
        Incrementally train the model
        """
        # we would be using the function `AuthorTopicModel.update` here (similar to model.update(corpus_new, author2doc_new))

        pass
```

- **Hierarchical Dirichlet Process Model**

```python
class SklearnWrapperHDPModel(models.HdpModel, TransformerMixin, BaseEstimator):
    """
    Base Hierarchical Dirichlet Process module
    """

    def __init__(self, corpus, id2word, max_chunks=None, max_time=None,
            chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1,
            gamma=1, eta=0.01, scale=1.0, var_converge=0.0001,
            outputdir=None, random_state=None):
        """
        Initialize the Hierarchical Dirichlet Process model using the parameter values passed
        """
        # If a training corpus was provided, start estimating the model right away
        # If not given, the model is left untrained (probably because the user
        # wants to call the `update` method manually later).

        pass


    def get_params(self, deep=True):
        """
        Return all the relevant parameters of the model as a dictionary
        """
        pass


    def set_params(self, **parameters):
        """
        Set all the parameters passed
        """
        for parameter, value in parameters.items():
            self.parameter = value
        return self


    def fit(self, X, y=None):
        """
        Fit the input data into the class object
        Call gensim.models.HdpModel
        """
        # set the corpus and id2word using the parameter passed

        models.HdpModel.__init__(self, corpus=self.corpus, id2word=self.id2word, max_chunks=self.max_chunks,
            max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau, K=self.K, T=self.T,
            alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale, var_converge=self.var_converge,
            outputdir=self.outputdir, random_state=self.random_state)

        return self

    def transform(self, docs):
        """
        Return the topics corresponding to documents passed as the parameter
        """
        # We could infer topic distributions on the documents passed similar to model[doc_bow]

        pass


    def partial_fit(self, data):
        """
        Incrementally train the model
        """
        # we would be using the function `HdpModel.update` here (similar to model.update(corpus))

        pass
```
