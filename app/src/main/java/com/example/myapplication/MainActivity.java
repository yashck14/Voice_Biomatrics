package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Random;

public class MainActivity extends AppCompatActivity{

    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private AudioRecorder audioRecorder,TestAudioRecorder;
    private boolean permissionToRecordAccepted = false;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};



    //FOR HMM
    private int numStates=5;
    private int numObservations=27;
    private double[][] M_transitionProb;
    private double[][] M_emissionProb;
    private double[] M_initialProb;


    private HMM hmm;
    private HMMTrainer hmmTrainer;
    private HMMPredictor hmmPredictor;
    double a,a1;
    double b,b1;
    double c,c1;
    double d,d1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
        String AudioData = getExternalFilesDir(null).getAbsolutePath() + "/audio1.wav";
        String TestdataAudio = getExternalFilesDir(null).getAbsolutePath() + "/audio0.wav";
        DecimalFormat df = new DecimalFormat("#.########");
        df.setRoundingMode(RoundingMode.CEILING);


        audioRecorder = new AudioRecorder(AudioData);
        TestAudioRecorder = new AudioRecorder(TestdataAudio);

        hmm = new HMM(5, 27); // Example: 5 states, 13 observation symbols (for MFCC features)
//        hmmTrainer = new HMMTrainer(hmm);
//        hmmPredictor = new HMMPredictor(hmm);
        M_transitionProb = new double[numStates][numStates];
        M_emissionProb = new double[numStates][numObservations];
        M_initialProb = new double[numStates];
        initializeProbabilities();


        Button startButton = findViewById(R.id.startRecordingButton);
        Button stopButton = findViewById(R.id.stopRecordingButton);
        Button check = findViewById(R.id.check);
        TextView textView_train_b1 = findViewById(R.id.textView3);
        TextView textView_train_c1 = findViewById(R.id.textView4);
        TextView textView_train_d1 = findViewById(R.id.textView5);
        TextView textView_train_b = findViewById(R.id.textView6);
        TextView textView_train_c = findViewById(R.id.textView7);
        TextView textView_train_d = findViewById(R.id.textView8);


        Button startTestButton = findViewById(R.id.startTestRecordingButton);
        Button stopTestButton = findViewById(R.id.stopTestRecordingButton);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (permissionToRecordAccepted) {
                    audioRecorder.startRecording();
                    Toast.makeText(MainActivity.this, "Recording started", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(MainActivity.this, "Permission to record denied", Toast.LENGTH_SHORT).show();
                }
            }
        });

        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                audioRecorder.stopRecording();
                MFCCExtractor.feature(AudioData);
                List<double[]> mfccFeaturesList = MFCCExtractor.extractMFCC();
                List<double[]> lpcFeaturesList = MFCCExtractor.extractLPC();
                List<double[]> spectralFeaturesList = MFCCExtractor.extractSpectralFeatures();

                int[] observations = flattenFeatures(mfccFeaturesList, lpcFeaturesList, spectralFeaturesList);

                train(observations, 250); // Train the model with the observation sequences
                Toast.makeText(MainActivity.this, "Recording stopped", Toast.LENGTH_SHORT).show();



                int[] prediction = predict(observations); // Predict the state sequence // Predict the state sequences
                a = computeLikelihood(prediction);
                b = forwardBackwardScore(observations);
                c= score_1(observations);
                d= viterbiScore(observations);
                textView_train_b.setVisibility(View.VISIBLE);
                textView_train_c.setVisibility(View.VISIBLE);
                textView_train_d.setVisibility(View.VISIBLE);

                textView_train_b.setText(Double.toString(b));
                textView_train_c.setText(Double.toString(c));
                textView_train_d.setText(Double.toString(d));


            }
        });
        startTestButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                TestAudioRecorder.startRecording();
                Toast.makeText(MainActivity.this, "Test Recording started", Toast.LENGTH_SHORT).show();
            }
        });
        stopTestButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                TestAudioRecorder.stopRecording();
                MFCCExtractor.feature(TestdataAudio);
                List<double[]> mfccFeaturesList = MFCCExtractor.extractMFCC();
                List<double[]> lpcFeaturesList = MFCCExtractor.extractLPC();
                List<double[]> spectralFeaturesList = MFCCExtractor.extractSpectralFeatures();

                int[] observations = flattenFeatures(mfccFeaturesList, lpcFeaturesList, spectralFeaturesList);
                int[] prediction = predict(observations); // Predict the state sequence // Predict the state sequences
                a1 = computeLikelihood(prediction);
                b1 = forwardBackwardScore(observations);
                c1 = score_1(observations);
                d1 = viterbiScore(observations);
                textView_train_b1.setVisibility(View.VISIBLE);
                textView_train_c1.setVisibility(View.VISIBLE);
                textView_train_d1.setVisibility(View.VISIBLE);
                textView_train_b1.setText(String.valueOf(b1));
                textView_train_c1.setText(String.valueOf(c1));
                textView_train_d1.setText(String.valueOf(d1));

                Toast.makeText(MainActivity.this, "Recording stopped", Toast.LENGTH_SHORT).show();
                Log.d("LIKELYHOOD",String.valueOf(a1)+" Predection 2:- "+String.valueOf(b1)+" Predection 3:- "+String.valueOf(c1)+" Predection 4:- "+String.valueOf(d1));

            }
        });
        check.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Log.d("LIKELYHOOD",String.valueOf(a)+" Predection 2:- "+String.valueOf(b)+" Predection 3:- "+String.valueOf(c)+" Predection 4:- "+String.valueOf(d));
                
            }
        });

    }

    //DATA PROCESSING
    private int[] flattenFeatures(List<double[]> mfccFeatures, List<double[]> lpcFeatures, List<double[]> spectralFeatures) {
        int totalLength = mfccFeatures.size() * mfccFeatures.get(0).length + lpcFeatures.size() * lpcFeatures.get(0).length + spectralFeatures.size()*2;
        int[] flattenedFeatures = new int[totalLength];

        double minValue_1 = Double.MAX_VALUE;
        double maxValue_1 = Double.MIN_VALUE;
        double minValue_2 = Double.MAX_VALUE;
        double maxValue_2 = Double.MIN_VALUE;
        double minValue_3 = Double.MAX_VALUE;
        double maxValue_3 = Double.MIN_VALUE;

        // Find min and max values for normalization
        for (double[] mfcc : mfccFeatures) {
            for (double val : mfcc) {
                if (val < minValue_1) minValue_1 = val;
                if (val > maxValue_1) maxValue_1 = val;
            }
        }

        for (double[] lpc : lpcFeatures) {
            for (double val : lpc) {
                if (val < minValue_2) minValue_2 = val;
                if (val > maxValue_2) maxValue_2 = val;
            }
        }

        for (double[] spectral : spectralFeatures) {
            for (double val : spectral) {
                if (val < minValue_3) minValue_3 = val;
                if (val > maxValue_3) maxValue_3 = val;
            }
        }

        // Normalize and discretize features
        int index = 0;
        for (double[] mfcc : mfccFeatures) {
            for (double val : mfcc) {
                flattenedFeatures[index++] = (int) discretize(val, minValue_1, maxValue_1, numObservations);
            }
        }

        for (double[] lpc : lpcFeatures) {
            for (double val : lpc) {
                flattenedFeatures[index++] = (int) discretize(val, minValue_2, maxValue_2, numObservations);
            }
        }

        for (double[] spectral : spectralFeatures) {
            for (double val : spectral) {
                flattenedFeatures[index++] = (int) discretize(val, minValue_3, maxValue_3, numObservations);
            }
        }

        return flattenedFeatures;
    }

    private double discretize(double value, double minValue, double maxValue, int numObservations) {
        // Normalize value to [0, 1]
        double normalizedValue = (value - minValue) / (maxValue - minValue);

        // Scale to the number of observations
        return (normalizedValue * (numObservations - 1));
    }
    private double computeLikelihood(int[] prediction) {
        // Simplified likelihood computation
        // In practice, use forward algorithm or other methods to compute actual likelihood
        double likelihood = 0.0;
        for (int state : prediction) {
            likelihood += M_initialProb[state];
        }
        return likelihood / prediction.length;
    }

    //DATA PROCESSIMG END

    //HMM DECLARING DATA
    private void initializeProbabilities() {
        Random random = new Random();

        // Initialize transition probabilities
        for (int i = 0; i < numStates; i++) {
            double sum = 0;
            for (int j = 0; j < numStates; j++) {
                M_transitionProb[i][j] = random.nextDouble()*10+1.3;
                sum += M_transitionProb[i][j];
            }
            for (int j = 0; j < numStates; j++) {
                M_transitionProb[i][j] /= sum;
            }
        }

        // Initialize emission probabilities
        for (int i = 0; i < numStates; i++) {
            double sum = 0;
            for (int j = 0; j < numObservations; j++) {
                M_emissionProb[i][j] = random.nextDouble()*10+1.3;
                sum += M_emissionProb[i][j];
            }
            for (int j = 0; j < numObservations; j++) {
                M_emissionProb[i][j] /= sum;
            }
        }

        // Initialize initial state probabilities
        double sum = 0;
        for (int i = 0; i < numStates; i++) {
            M_initialProb[i] = random.nextDouble()*10+1.3;
            sum += M_initialProb[i];
        }
        for (int i = 0; i < numStates; i++) {
            M_initialProb[i] /= sum;
        }
    }
    //HMM DECLARING DATA END

    //TRAIN MODEL
    public void train(int[] observations, int maxIterations) {
        int numStates = M_transitionProb.length;
        int numObservations = M_emissionProb[0].length;
        int T = observations.length;

        System.out.println("Number of States: " + numStates);
        System.out.println("Number of Observations: " + numObservations);
        System.out.println("Length of Observation Sequence: " + T);

        double[][] transitionProb = M_transitionProb;
        double[][] emissionProb = M_emissionProb;
        double[] initialProb = M_initialProb;
        double[][] alpha;
        double[][] beta;
        double[][] gamma;
        double[][][] xi;

        for (int it = 0; it < maxIterations; it++) {
            alpha = forward(observations, numStates, T, transitionProb, emissionProb, initialProb);
            beta = backward(observations, numStates, T, transitionProb, emissionProb);

            gamma = new double[T][numStates];
            xi = new double[T - 1][numStates][numStates];

            for (int t = 0; t < T - 1; t++) {
                double sumGamma = 0;
                for (int i = 0; i < numStates; i++) {
                    gamma[t][i] = alpha[t][i] * beta[t][i];
                    sumGamma += gamma[t][i];
                }
                for (int i = 0; i < numStates; i++) {
                    gamma[t][i] /= sumGamma;
                }

                for (int i = 0; i < numStates; i++) {
                    for (int j = 0; j < numStates; j++) {
                        xi[t][i][j] = alpha[t][i] * transitionProb[i][j] * emissionProb[j][observations[t + 1]] * beta[t + 1][j];
                        xi[t][i][j] /= sumGamma;
                    }
                }
            }

            double[] gammaSum = new double[numStates];
            for (int i = 0; i < numStates; i++) {
                gammaSum[i] = 0;
                for (int t = 0; t < T - 1; t++) {
                    gammaSum[i] += gamma[t][i];
                }
            }

            for (int i = 0; i < numStates; i++) {
                initialProb[i] = gamma[0][i]+0.135;
            }

            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numStates; j++) {
                    double num = 0;
                    for (int t = 0; t < T - 1; t++) {
                        num += xi[t][i][j];
                    }
                    transitionProb[i][j] = num / gammaSum[i]+0.135;
                }
            }

            for (int j = 0; j < numStates; j++) {
                for (int k = 0; k < numObservations; k++) {
                    double num = 0;
                    for (int t = 0; t < T; t++) {
                        if (observations[t] == k) {
                            num += gamma[t][j];
                        }
                    }
                    emissionProb[j][k] = num / gammaSum[j]+0.135;
                }
            }
        }
        System.out.println("\n after processing transitionProb:-\n");
        for (int i = 0;i<5;i++){
            for (int j =0;j<5;j++){
                System.out.print(transitionProb[i][j]);
            }
        }System.out.println("\nafter processing emissionProb:-\n");
        for (int i = 0;i<5;i++){
            for (int j =0;j<27;j++){
                System.out.print(emissionProb[i][j]);
            }
        }
        System.out.println("\nafter processing OBSERVATIONS:-\n");
        for (int i =0;i< 5;i++){
            System.out.print(initialProb[i]);
        }
            M_emissionProb = emissionProb;
            M_transitionProb = transitionProb;
            M_initialProb = initialProb;
    }

    private double[][] forward(int[] observations, int numStates, int T, double[][] transitionProb, double[][] emissionProb, double[] initialProb) {
        double[][] alpha = new double[T][numStates];

        // Initialization
        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = initialProb[i] * emissionProb[i][observations[0]];
            if (Double.isNaN(alpha[0][i])) {
                alpha[0][i] = 0;
            }
        }

        // Forward procedure
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * transitionProb[j][i];
                }
                alpha[t][i] *= emissionProb[i][observations[t]];
                if (Double.isNaN(alpha[t][i])) {
                    alpha[t][i] = 0;
                }
            }
        }

        return alpha;
    }


    private double[][] backward(int[] observations, int numStates, int T, double[][] transitionProb, double[][] emissionProb) {
        double[][] beta = new double[T][numStates];

        // Initialization
        for (int i = 0; i < numStates; i++) {
            beta[T - 1][i] = 1;
        }

        // Backward procedure
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < numStates; i++) {
                beta[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    beta[t][i] += beta[t + 1][j] * transitionProb[i][j] * emissionProb[j][observations[t + 1]];
                    if (Double.isNaN(beta[t][i])) {
                        beta[t][i] = 0;
                    }
                }
            }
        }

        return beta;
    }

    //TRAIN MODEL END

    //PREDICT MODEL
    public int[] predict(int[] observations) {
        int numStates = M_transitionProb.length;
        int T = observations.length;
        double[][] delta = new double[T][numStates];
        int[][] psi = new int[T][numStates];

        for (int i = 0; i < numStates; i++) {
            delta[0][i] = M_initialProb[i] * M_emissionProb[i][observations[0]];
            psi[0][i] = 0;
        }
        // Recursion
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                double maxDelta = -1;
                int maxPsi = -1;
                for (int j = 0; j < numStates; j++) {
                    double value = delta[t - 1][j] * M_transitionProb[j][i];
                    if (value > maxDelta) {
                        maxDelta = value;
                        maxPsi = j;
                    }
                }

                delta[t][i] = maxDelta * M_emissionProb[i][observations[t]];
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

        for (int t = T - 2; t > 0; t--) {
            path[t] = psi[t + 1][path[t + 1]];
        }

        return path;
    }

    public double score_1(int[] observations) {
        int numStates = M_transitionProb.length;
        int T = observations.length;

        double[][] alpha = new double[T][numStates];

        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = M_initialProb[i] * M_emissionProb[i][observations[0]];
        }

        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * M_transitionProb[j][i];
                }
                alpha[t][i] *= M_emissionProb[i][observations[t]];
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
        int numStates = M_transitionProb.length;
        int T = observations.length;

        // Forward probabilities
        double[][] alpha = new double[T][numStates];
        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = M_initialProb[i] * M_emissionProb[i][observations[0]];
        }
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * M_transitionProb[j][i];
                }
                alpha[t][i] *= M_emissionProb[i][observations[t]];
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
                    beta[t][i] += beta[t + 1][j] * M_transitionProb[i][j] * M_emissionProb[j][observations[t + 1]];
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
        int numStates = M_transitionProb.length;
        int T = observations.length;
        for (int i =0;i< observations.length;i++){
            System.out.println(observations[i]+",");
        }

        double[][] delta = new double[T][numStates];
        int[][] psi = new int[T][numStates];

        // Initialization step
        for (int i = 0; i < numStates; i++) {
            delta[0][i] = M_initialProb[i] * M_emissionProb[i][observations[0]];
            psi[0][i] = 0;
        }

        // Recursion step
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                double maxDelta = -1;
                int maxPsi = -1;
                for (int j = 0; j < numStates; j++) {
                    double value = delta[t - 1][j] * M_transitionProb[j][i];
                    if (value > maxDelta) {
                        maxDelta = value;
                        maxPsi = j;
                    }
                }
                delta[t][i] = maxDelta * M_emissionProb[i][observations[t]];
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

    //PREDICT MODEL END



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        permissionToRecordAccepted = requestCode == REQUEST_RECORD_AUDIO_PERMISSION && grantResults[0] == PackageManager.PERMISSION_GRANTED;
    }


}

//vatsal vs vatsal: LIKELYHOOD: 0.33500000000000074 Predection 2:- 2.9306318798348946E-60 Predection 3:- 0.6095085377366066 Predection 4:- 1.2764702113845474E-154
// vatsal vs vatsal:- LIKELYHOOD: 0.33500000000000074 Predection 2:- 2.3808039837076152E-70 Predection 3:- 0.3813061391804402 Predection 4:- 1.391819568141048E-183
//vatsal vs gaurav:- LIKELYHOOD: 0.33500000000000074 Predection 2:- 3.3688521345978695E-98 Predection 3:- 0.3800493527570013 Predection 4:- 3.5478153858701257E-249
//vatsal vs nikhil:- LIKELYHOOD: 0.33500000000000074 Predection 2:- 5.277316163974374E-87 Predection 3:- 0.38137708298797657 Predection 4:- 4.140781306383469E-219