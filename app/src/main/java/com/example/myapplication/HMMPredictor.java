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
            if (Double.isNaN(delta[0][i]) || Double.isInfinite(delta[0][i])) {
                System.out.println("Invalid initial delta value at state " + i + ": " + delta[0][i]);
                System.out.println("Initial Prob: " + hmm.getInitialProb()[i]);
                System.out.println("Emission Prob: " + hmm.getEmissionProb()[i][observations[0]]);
            }
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
                if (Double.isNaN(delta[t][i]) || Double.isInfinite(delta[t][i])) {
                    System.out.println("Invalid delta value at time " + t + " state " + i + ": " + delta[t][i]);
                }
                if (maxPsi == -1) {
                    System.out.println("Invalid psi value at time " + t + " state " + i);
                }
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
        System.out.println("Delta:");
        for (int t = 0; t < T; t++) {
            System.out.println(Arrays.toString(delta[t]));
        }

        System.out.println("Psi:");
        for (int t = 0; t < T; t++) {
            System.out.println(Arrays.toString(psi[t]));
        }
        System.out.println("Initial path state: " + path[T - 1]);
        for (int t = T - 2; t > 0; t--) {
            path[t] = psi[t + 1][path[t + 1]];
            System.out.println("Backtracking path[" + t + "] = " + path[t]);
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
        for (int i = 0; i < numStates; i++) {
            score += alpha[T - 1][i];
        }

        return score;
    }
}
