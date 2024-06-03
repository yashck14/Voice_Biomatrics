package com.example.myapplication;
import android.media.MediaRecorder;

import java.io.IOException;

public class AudioRecorder {
    private MediaRecorder recorder;
    private String fileName;

    public AudioRecorder(String fileName) {
        this.fileName = fileName;
    }

    public void startRecording() {
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.DEFAULT);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.DEFAULT);
        recorder.setOutputFile(fileName);

        try {
            recorder.prepare();
            recorder.start();
        } catch (IOException e) {
            e.printStackTrace();
            // Handle IOException properly
        } catch (IllegalStateException e) {
            e.printStackTrace();
            // Handle IllegalStateException properly
        }
    }

    public void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;
        } catch (RuntimeException e) {
            e.printStackTrace();
        }
    }
}
