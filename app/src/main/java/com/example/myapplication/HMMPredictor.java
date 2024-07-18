package com.example.myapplication;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class HMMPredictor {
    private HMM hmm;

    public HMMPredictor(HMM hmm) {
        this.hmm = hmm;
    }

    public int[] predict(int[] observations) {
        int numStates = hmm.getTransitionProb().length;
        int T = observations.length;

        double[][] delta = new double[T][numStates];
        int[][] psi = new int[T][numStates];

        for (int i = 0; i < numStates; i++) {
            delta[0][i] = hmm.getInitialProb()[i] * hmm.getEmissionProb()[i][observations[0]];
            psi[0][i] = 0;
        }
        // Recursion
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                double maxDelta = -1;
                int maxPsi = -1;
                for (int j = 0; j < numStates; j++) {
                    double value = delta[t - 1][j] * hmm.getTransitionProb()[j][i];
                    if (value > maxDelta) {
                        maxDelta = value;
                        maxPsi = j;
                    }
                }

                delta[t][i] = maxDelta * hmm.getEmissionProb()[i][observations[t]];
                psi[t][i] = maxPsi;
            }
        }

        int[] path = new int[T];
        double maxDelta = -1;
        for (int i = 0; i < numStates; i++) {
            if (delta[T - 1][i] > maxDelta) {
                maxDelta = delta[T - 1][i];
                path[T - 1] = i;
            }
        }
        return path;
    }

    public double score(int[] observations) {
        int numStates = hmm.getTransitionProb().length;
        int T = observations.length;

        double[][] alpha = new double[T][numStates];

        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = hmm.getInitialProb()[i] * hmm.getEmissionProb()[i][observations[0]];
        }

        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * hmm.getTransitionProb()[j][i];
                }
                alpha[t][i] *= hmm.getEmissionProb()[i][observations[t]];
            }
        }

        double score = 0;
        for (int j = 0;j<T-1;j++){
            for (int i = 0; i < numStates; i++) {
                score += alpha[j][i];
            }
        }

        return score;
    }
    public double forwardBackwardScore(int[] observations) {
        int numStates = hmm.getTransitionProb().length;
        int T = observations.length;

        // Forward probabilities
        double[][] alpha = new double[T][numStates];
        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = hmm.getInitialProb()[i] * hmm.getEmissionProb()[i][observations[0]];
        }
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * hmm.getTransitionProb()[j][i];
                }
                alpha[t][i] *= hmm.getEmissionProb()[i][observations[t]];
            }
        }

        // Backward probabilities
        double[][] beta = new double[T][numStates];
        for (int i = 0; i < numStates; i++) {
            beta[T - 1][i] = 1.0;
        }
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < numStates; i++) {
                beta[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    beta[t][i] += beta[t + 1][j] * hmm.getTransitionProb()[i][j] * hmm.getEmissionProb()[j][observations[t + 1]];
                }
            }
        }

        // Compute the likelihood of the observations
        double score = 0.0;
        for (int i = 0; i < numStates; i++) {
            score += alpha[T - 1][i] * beta[T - 1][i];
        }

        return score;
    }
    public double viterbiScore(int[] observations) {
        int numStates =hmm.getTransitionProb().length;
        int T = observations.length;
        for (int i =0;i< observations.length;i++){
            System.out.println(observations[i]+",");
        }

        double[][] delta = new double[T][numStates];
        int[][] psi = new int[T][numStates];

        // Initialization step
        for (int i = 0; i < numStates; i++) {
            delta[0][i] =hmm.getInitialProb()[i] * hmm.getEmissionProb()[i][observations[0]];
            psi[0][i] = 0;
        }

        // Recursion step
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                double maxDelta = -1;
                int maxPsi = -1;
                for (int j = 0; j < numStates; j++) {
                    double value = delta[t - 1][j] * hmm.getTransitionProb()[j][i];
                    if (value > maxDelta) {
                        maxDelta = value;
                        maxPsi = j;
                    }
                }
                delta[t][i] = maxDelta * hmm.getEmissionProb()[i][observations[t]];
                psi[t][i] = maxPsi;
            }
        }

        // Termination step
        double maxProbability = -1;
        for (int i = 0; i < numStates; i++) {
            if (delta[T - 1][i] > maxProbability) {
                maxProbability = delta[T - 1][i];
            }
        }

        return maxProbability;
    }
}
