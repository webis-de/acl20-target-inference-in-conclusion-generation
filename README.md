# Paper: Target Inference in Conclusion Generation

This is the code for the paper *Target Inference in Conclusion Generation*.

Milad Alshomary, Shahbaz Syed, Martin Potthast and Henning Wachstmuth


      @InProceedings{elbaff:2020,
        author =              {Milad Alshomary, Shahbaz Syed, Martin Potthast and Henning Wachstmuth},
        booktitle =           {The 58th annual meeting of the Association for Computational Linguistics (ACL) },
        month =               jul,
        publisher =           {ACL},
        site =                {Seattle, USA},
        title =               {{Target Inference in Conclusion Generation}},
        year =                2020
      }

-----------------------------------------------

## Preprocessing
All scripts for preprocessing the data are in the ``preprocessing`` folder.

## Target Identification
To tag targets in premises and conclusions, we train a sequence tagger on the IBM dataset. The code is in ``target_identification/claim_target_tagger.py``. A trained model ready to be used is under ``target_identification/models/target_tagger_model.pt``. The preprocessed IBM dataset that the model was trained over is under ``target_identification/data/ibm_ds``


## Target Inference
Preprocessed and tagged corpora is under ``target_inference/data`` along with the knowledge base of targets used in our approach.

### Ranking approach
The code for training ranking models is under ``target_ranking/ranking_targets.py``. The trained models are under ``target_inference/models``

### Target embedding learnin approach
- Code for training the triplet neural network is under ``target_inference/siamese-triplet``
- ``targets_inference_experiment.py`` contains all experiments performed for target inference.
