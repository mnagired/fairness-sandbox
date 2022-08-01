# Blum_Stangl Case Study

Empirical replication of results from [Blum_Stangl](https://arxiv.org/abs/1912.01094)

## Description

The above paper theoretically describes settings in which fairness-constrained optimization on biased data recovers the Bayes optimal classifier on the true data distribution.

In this case study, we specifically zoom into one of the findings of [the paper](https://arxiv.org/abs/2204.10233), that is, Equalized Odds constrained empirical risk minimization on data with underrepresentation bias can recover the Bayes optimal classifier on the unbiased data. This result requires several structural assumptions on the data generating process and we will draw on the described data generating procedure when simulating data for our simulation. 

## Authors + Collaborators

Lead Developers: Manish Nagireddy [@mnagired](http://twitter.com/mnagired) [(LinkedIn)](https://www.linkedin.com/in/mnagireddy/) and Nil-Jana Akpinar [(webpage)](http://nakpinar.github.io)

Primary Advisor: Hoda Heidari [(webpage)](https://www.cs.cmu.edu/~hheidari/)

Collaborators:
   *  Steven Wu [(webpage)](http://zstevenwu.com)
   *  Logan Stapleton [(webpage)](http://loganstapleton.com)
   *  Haiyi Zhu [(webpage)](http://haiyizhu.com)
   *  Hao-Fei Cheng [(webpage)](https://www-users.cse.umn.edu/~cheng635/)

## Codebase Structure

* `under_synthetic`: main notebook file which replicates the above result
* `plots.ipynb`: load csv data files and generate plots

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details


## Contact

Please reach out to Nil-Jana Akpinar [(email)](mailto:nakpinar@andrew.cmu.edu) with any comments/questions/suggestions!
