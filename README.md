# Bias Injection Sandbox

A framework which assesses the effectiveness of fairness-enhancing interventions. Full paper linked [here](https://arxiv.org/abs/2204.10233).

## Structure

`sandbox.ipynb`: main file to run the sandbox's functionalities

## Description

Our sandbox offers the following pipeline:

### Data
1. Upload Dataset
2. Choose existing dataset (e.g. Adult Income)
3. Generate Synthetic Dataset

### Model
1. Train any ml model of choice

### Bias Injection
Select one (or more) bias(es) to inject into the data from the following list:
1. Representation Bias (under-sampling subsets of the data)
2. Measurement Bias (adding noise)
3. Omitted Variable Bias
4. Label Noise Bias
5. Over-Sampling Bias
6. Under-Sampling Bias

### Fairness Intervention
Select one of the following interventions:
1. [Correlation Remover](https://fairlearn.org/v0.7.0/api_reference/fairlearn.preprocessing.html#module-fairlearn.preprocessing) (Pre-Processing)
2. [Exponentiated Gradient](https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.ExponentiatedGradient) (In-Processing)
3. [Grid Search](https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.GridSearch) (In-Processing)
4. [Threshold Optimizer](https://fairlearn.org/v0.7.0/api_reference/fairlearn.postprocessing.html) (Post-Processing)

### Fairness Visualization
After selecting a metric of your choice (e.g. accuracy, precision, roc_auc, etc), we output a plot which displays the effectiveness of the fairness intervention's ability to mitigate the bias you injected, with respect to the ground truth data.

## Authors + Collaborators

Lead Developer: Manish Nagireddy [@mnagired](http://twitter.com/mnagired) [(LinkedIn)](https://www.linkedin.com/in/mnagireddy/)

Primary Author: Nil-Jana Akpinar [(webpage)](http://nakpinar.github.io)

Primary Advisor: Hoda Heidari [(webpage)](https://www.cs.cmu.edu/~hheidari/)

Collaborators:
   *  Steven Wu [(webpage)](http://zstevenwu.com)
   *  Logan Stapleton [(webpage)](http://loganstapleton.com)
   *  Haiyi Zhu [(webpage)](http://haiyizhu.com)
   *  Hao-Fei Cheng [(webpage)](https://www-users.cse.umn.edu/~cheng635/)

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details

## Contact

Please reach out to Nil-Jana Akpinar [(email)](mailto:nakpinar@andrew.cmu.edu) with any comments/questions/suggestions!
