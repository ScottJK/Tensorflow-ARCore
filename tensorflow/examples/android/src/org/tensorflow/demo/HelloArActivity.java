/*
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.MotionEvent;
import android.widget.Toast;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Point;
import com.google.ar.core.Point.OrientationMode;
import com.google.ar.core.PointCloud;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.rendering.BackgroundRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer.BlendMode;
import com.google.ar.core.examples.java.common.rendering.PlaneRenderer;
import com.google.ar.core.examples.java.common.rendering.PointCloudRenderer;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import org.tensorflow.demo.env.ImageUtils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import javax.vecmath.Vector3f;

import utilities.CoordinateTransformation;

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application will display any detected planes and will allow the user to tap on a
 * plane to place a 3d model of the Android robot.
 */
public class HelloArActivity extends AppCompatActivity implements GLSurfaceView.Renderer {
    private static final String TAG = HelloArActivity.class.getSimpleName();
    // my new stuff
    //
    //    // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
    //    // must be manually placed in the assets/ directory by the user.
    //    // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
    //    // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
    //    // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;
    // ARCore full resolution GL texture typically has a size of 1920 x 1080.
    private static final int TEXTURE_WIDTH = 1920;
    private static final int TEXTURE_HEIGHT = 1080;
    // We choose a lower sampling resolution.
    private static final int IMAGE_WIDTH = 1280;
    private static final int IMAGE_HEIGHT = 720;
    private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
    private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
    private final ObjectRenderer virtualObject = new ObjectRenderer();
    private final ObjectRenderer virtualObjectShadow = new ObjectRenderer();
    private final PlaneRenderer planeRenderer = new PlaneRenderer();
    private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
    // Temporary matrix allocated here to reduce number of allocations for each frame.
    private final float[] anchorMatrix = new float[16];
    // Anchors created from taps used for object placing.
    private final ArrayList<Anchor> anchors = new ArrayList<>();
    private final float[] bestMatchCenter = new float[2];
    // The fields below are used for the GPU_DOWNLOAD image acquisition path.
    private final TextureReader textureReader = new TextureReader();
    private final ObjectRenderer arrowObject = new ObjectRenderer();
    private final float[] projmtx = new float[16];
    private final float[] viewmtx = new float[16];
    private final float[] arrowMatrix = new float[16];
    private final float MIN_DISTANCE_BETWEEN_OBJECTS = 1.0f;
    // Rendering. The Renderers are created here, and initialized when the GL surface is created.
    private GLSurfaceView surfaceView;
    private boolean installRequested;
    private Session session;
    private DisplayRotationHelper displayRotationHelper;
    private TapHelper tapHelper;
    private Classifier detector;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private Matrix cropToScreenTransform;
    private boolean bFirstFrame = true;
    private Classifier.Recognition bestMatch = null;
    private boolean detectingObjects = false;
    private int gpuDownloadFrameBufferIndex = -1;
    private long lastFrameTime;
    private Handler handler;
    private HandlerThread handlerThread;

    private static float getPoseSquareDistance(Pose p1, Pose p2) {
        float dx = p1.tx() - p2.tx();
        float dy = p1.ty() - p2.ty();
        float dz = p1.tz() - p2.tz();

        // Compute the straight-line distance.
        float distanceMeters = (dx * dx + dy * dy + dz * dz);
        return distanceMeters;
    }

    private static float square(float f) {
        return f * f;
    }

    private static double square(double f) {
        return f * f;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        surfaceView = findViewById(R.id.surfaceview);
        displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

        // Set up tap listener.
        tapHelper = new TapHelper(/*context=*/ this);
        surfaceView.setOnTouchListener(tapHelper);

        // Set up renderer.
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        installRequested = true;

        detector =
                TensorFlowYoloDetector.create(
                        getAssets(),
                        YOLO_MODEL_FILE,
                        YOLO_INPUT_SIZE,
                        YOLO_INPUT_NAME,
                        YOLO_OUTPUT_NAMES,
                        YOLO_BLOCK_SIZE);
    }

    @Override
    public synchronized void onResume() {
        Log.d(TAG, "onResume " + this);
        super.onResume();
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        if (session == null) {
            Exception exception = null;
            String message = null;
            try {
                switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    case INSTALL_REQUESTED:
                        installRequested = true;
                        return;
                    case INSTALLED:
                        break;
                }

                // ARCore requires camera permissions to operate. If we did not yet obtain runtime
                // permission on Android M and above, now is a good time to ask the user for it.
                if (!CameraPermissionHelper.hasCameraPermission(this)) {
                    CameraPermissionHelper.requestCameraPermission(this);
                    return;
                }

                // Create the session.
                session = new Session(/* context= */ this);

            } catch (UnavailableArcoreNotInstalledException
                    | UnavailableUserDeclinedInstallationException e) {
                message = "Please install ARCore";
                exception = e;
            } catch (UnavailableApkTooOldException e) {
                message = "Please update ARCore";
                exception = e;
            } catch (UnavailableSdkTooOldException e) {
                message = "Please update this app";
                exception = e;
            } catch (UnavailableDeviceNotCompatibleException e) {
                message = "This device does not support AR";
                exception = e;
            } catch (Exception e) {
                message = "Failed to create AR session";
                exception = e;
            }

            if (message != null) {
                messageSnackbarHelper.showError(this, message);
                Log.e(TAG, "Exception creating session", exception);
                return;
            }
        }

        // Note that order matters - see the note in onPause(), the reverse applies here.
        try {
            session.resume();
        } catch (CameraNotAvailableException e) {
            // In some cases (such as another camera app launching) the camera may be given to
            // a different app instead. Handle this properly by showing a message and recreate the
            // session at the next iteration.
            messageSnackbarHelper.showError(this, "Camera not available. Please restart the app.");
            session = null;
            return;
        }

        surfaceView.onResume();
        displayRotationHelper.onResume();

        messageSnackbarHelper.showMessage(this, "Searching for surfaces...");
    }

    @Override
    public synchronized void onPause() {
        Log.d(TAG, "onPause " + this);

        if (!isFinishing()) {
            Log.d(TAG, "Requesting finish");
            finish();
        }

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception!", e);
        }

        super.onPause();
        if (session != null) {
            // Note that the order matters - GLSurfaceView is paused first so that it does not try
            // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
            // still call session.update() and get a SessionPausedException.
            displayRotationHelper.onPause();
            surfaceView.onPause();
            session.pause();
        }
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
            Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
                    .show();
            if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
                // Permission denied with checking "Do not ask again".
                CameraPermissionHelper.launchPermissionSettings(this);
            }
            finish();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
        try {
            // Create the texture and pass it to ARCore session to be filled during update().
            backgroundRenderer.createOnGlThread(/*context=*/ this);
            planeRenderer.createOnGlThread(/*context=*/ this, "models/trigrid.png");
            pointCloudRenderer.createOnGlThread(/*context=*/ this);

            virtualObject.createOnGlThread(/*context=*/ this, "models/andy.obj", "models/andy.png");
            virtualObject.setMaterialProperties(0.0f, 2.0f, 0.5f, 6.0f);

            virtualObjectShadow.createOnGlThread(
                    /*context=*/ this, "models/andy_shadow.obj", "models/andy_shadow.png");
            virtualObjectShadow.setBlendMode(BlendMode.Shadow);
            virtualObjectShadow.setMaterialProperties(1.0f, 0.0f, 0.0f, 1.0f);

            arrowObject.createOnGlThread(/*context=*/ this, "models/arrow.obj", "models/green.png");
            arrowObject.setMaterialProperties(0.0f, 2.0f, 0.5f, 6.0f);

            // The image format can be either IMAGE_FORMAT_RGBA or IMAGE_FORMAT_I8.
            // Set keepAspectRatio to false so that the output image covers the whole viewport.
            textureReader.create(
                    /* context= */ this,
                    TextureReaderImage.IMAGE_FORMAT_RGBA,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                    false);

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            IMAGE_WIDTH, IMAGE_HEIGHT,
                            YOLO_INPUT_SIZE, YOLO_INPUT_SIZE,
                            90 - displayRotationHelper.getRotation(), true);

            cropToFrameTransform = new android.graphics.Matrix();
            frameToCropTransform.invert(cropToFrameTransform);

            cropToScreenTransform =
                    ImageUtils.getTransformationMatrix(
                            IMAGE_WIDTH, IMAGE_HEIGHT,
                            YOLO_INPUT_SIZE, YOLO_INPUT_SIZE,
                            90 - displayRotationHelper.getRotation(), true);
            cropToScreenTransform.invert(cropToScreenTransform);
            cropToScreenTransform.postRotate(90, IMAGE_HEIGHT / 2, IMAGE_HEIGHT / 2);

        } catch (IOException e) {
            Log.e(TAG, "Failed to read an asset file", e);
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        displayRotationHelper.onSurfaceChanged(width, height);
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        // Clear screen to notify driver it should not load any pixels from previous frame.
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        if (session == null) {
            return;
        }
        // Notify ARCore session that the view size changed so that the perspective matrix and
        // the video background can be properly adjusted.
        displayRotationHelper.updateSessionIfNeeded(session);

        try {
            session.setCameraTextureName(backgroundRenderer.getTextureId());

            // Obtain the current frame from ARSession. When the configuration is set to
            // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
            // camera framerate.
            Frame frame = session.update();
            Camera camera = frame.getCamera();

            // Handle taps. Handling only one tap per frame, as taps are usually low frequency
            // compared to frame rate.

            MotionEvent tap = tapHelper.poll();
            if (tap != null && camera.getTrackingState() == TrackingState.TRACKING) {
                for (HitResult hit : frame.hitTest(tap)) {
                    // Check if any plane was hit, and if it was hit inside the plane polygon
                    Trackable trackable = hit.getTrackable();
                    // Creates an anchor if a plane or an oriented point was hit.
                    if ((trackable instanceof Plane && ((Plane) trackable).isPoseInPolygon(hit.getHitPose()))
                            || (trackable instanceof Point
                            && ((Point) trackable).getOrientationMode()
                            == OrientationMode.ESTIMATED_SURFACE_NORMAL)) {
                        // Hits are sorted by depth. Consider only closest hit on a plane or oriented point.
                        // Cap the number of objects created. This avoids overloading both the
                        // rendering system and ARCore.
                        if (anchors.size() >= 20) {
                            anchors.get(0).detach();
                            anchors.remove(0);
                        }
                        // Adding an Anchor tells ARCore that it should track this position in
                        // space. This anchor is created on the Plane to place the 3D model
                        // in the correct position relative both to the world and to the plane.
                        anchors.add(hit.createAnchor());
                        break;
                    }
                }
            }

            // Draw background.
            backgroundRenderer.draw(frame);
//            double fps = 1.0 / (((double) (frame.getTimestamp() - lastFrameTime)) / 1000000000);
//            Log.i(TAG, "FPS: " + fps);
//            lastFrameTime = frame.getTimestamp();

            // If not tracking, don't draw 3d objects.
            if (camera.getTrackingState() == TrackingState.PAUSED) {
                return;
            }

            // Get projection matrix.
            camera.getProjectionMatrix(projmtx, 0, 0.1f, 100.0f);

            // Get camera matrix and draw.
            camera.getViewMatrix(viewmtx, 0);

            // Compute lighting from average intensity of the image.
            // The first three components are color scaling factors.
            // The last one is the average pixel intensity in gamma space.
            final float[] colorCorrectionRgba = new float[4];
            frame.getLightEstimate().getColorCorrection(colorCorrectionRgba, 0);

            // Make sure there aren't too many anchors
            if (session.getAllAnchors().size() > 300) {
                Log.w(TAG, "Too many anchors! Not running onDrawFrame thread anymore.");
                return;
            }

            // Visualize tracked points.
            PointCloud pointCloud = frame.acquirePointCloud();
            pointCloudRenderer.update(pointCloud);
            pointCloudRenderer.draw(viewmtx, projmtx);


            // Application is responsible for releasing the point cloud resources after
            // using it.
            pointCloud.release();

            // Check if we detected at least one plane. If so, hide the loading message.
            if (messageSnackbarHelper.isShowing()) {
                for (Plane plane : session.getAllTrackables(Plane.class)) {
                    if (plane.getType() == Plane.Type.HORIZONTAL_UPWARD_FACING
                            && plane.getTrackingState() == TrackingState.TRACKING) {
                        messageSnackbarHelper.hide(this);
                        break;
                    }
                }
            }

            if ((camera.getTrackingState() == TrackingState.TRACKING)) {

                if (anchors.size() >= 10) {
                    anchors.get(1).detach();
                    anchors.remove(1);
                }

                if (bestMatch != null && !detectingObjects) {
                    ArrayList<Vector3f> points = getPointCloudCoordinates(frame, 0.5f);
                    Log.d(TAG, "" + bestMatchCenter[0] + ", " + bestMatchCenter[1]);
                    if (points.size() > 0) {
                        Pose closestPose = null;
                        double smallestSqrDistance = Double.MAX_VALUE;
                        for (Vector3f v : points) {
                            Pose p = Pose.makeTranslation(v.x, v.y, v.z);
                            double[] pointCloud2dLocation = CoordinateTransformation.world2Screen(p, IMAGE_HEIGHT, IMAGE_WIDTH, viewmtx, projmtx);
                            if (!bestMatch.getLocation().contains((float) pointCloud2dLocation[0], (float) pointCloud2dLocation[1])) {
                                continue;
                            }
                            double sqrDistance = (bestMatchCenter[0] - pointCloud2dLocation[0]) * (bestMatchCenter[0] - pointCloud2dLocation[0]) + (bestMatchCenter[1] - pointCloud2dLocation[1]) * (bestMatchCenter[1] - pointCloud2dLocation[1]);
                            if (sqrDistance < smallestSqrDistance) {
                                smallestSqrDistance = sqrDistance;
                                closestPose = p;
                            }
                        }
                        if (closestPose != null) {
                            Float distanceToNearestPose = getDistanceToClosestAnchor(closestPose);
                            if (distanceToNearestPose == null || distanceToNearestPose > square(MIN_DISTANCE_BETWEEN_OBJECTS))
                                anchors.add(session.createAnchor(closestPose));
                        } else {
                            Log.w(TAG, "Ignored anchor because it was too far away from the center of the object!");
                        }
                    }
                    bestMatch = null;
                }
                renderProcessedImageGpuDownload();
                Log.d(TAG, "Anchors: " + session.getAllAnchors().size());
            }
            // Visualize arrow
            Anchor closestAnchor = getClosestAnchor(camera.getPose());
            if (closestAnchor != null) {
                closestAnchor.getPose().toMatrix(arrowMatrix, 0);
                arrowObject.updateModelMatrix(arrowMatrix, 0.15f);
                arrowObject.draw(viewmtx, projmtx, colorCorrectionRgba);
            }

            // Visualize planes.
            planeRenderer.drawPlanes(
                    session.getAllTrackables(Plane.class), camera.getDisplayOrientedPose(), projmtx);

            // Visualize anchors created by touch.
            float scaleFactor = 0.5f;
            for (Anchor anchor : anchors) {
                if (anchor.getTrackingState() != TrackingState.TRACKING) {
                    continue;
                }
                // Get the current pose of an Anchor in world space. The Anchor pose is updated
                // during calls to session.update() as ARCore refines its estimate of the world.
                anchor.getPose().toMatrix(anchorMatrix, 0);

                // Update and draw the model and its shadow.
                virtualObject.updateModelMatrix(anchorMatrix, scaleFactor);
                virtualObjectShadow.updateModelMatrix(anchorMatrix, scaleFactor);
                virtualObject.draw(viewmtx, projmtx, colorCorrectionRgba);
                virtualObjectShadow.draw(viewmtx, projmtx, colorCorrectionRgba);
            }

        } catch (Throwable t) {
            // Avoid crashing the application due to unhandled exceptions.
            Log.e(TAG, "Exception on the OpenGL thread", t);
        }
    }

    private void renderProcessedImageGpuDownload() {
        // If there is a frame being requested previously, acquire the pixels and process it.
        if (gpuDownloadFrameBufferIndex >= 0) {
            TextureReaderImage image = textureReader.acquireFrame(gpuDownloadFrameBufferIndex);
            if (image.format != TextureReaderImage.IMAGE_FORMAT_RGBA) {
                throw new IllegalArgumentException(
                        "Expected image in RGBA format, got format " + image.format);
            }
            ByteBuffer processedImageBytesGrayscale = image.buffer;
            // You should always release frame buffer after using. Otherwise the next call to
            // submitFrame() may fail.
            textureReader.releaseFrame(gpuDownloadFrameBufferIndex);
            if (!detectingObjects) {
                detectingObjects = true;
                Bitmap bm = Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888, true);
                bm.copyPixelsFromBuffer(processedImageBytesGrayscale);
                final Bitmap resizedBM = Bitmap.createBitmap(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, Bitmap.Config.ARGB_8888);
//            Bitmap resizedBM = Bitmap.createScaledBitmap(bm, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, false);
                final Canvas canvas = new Canvas(resizedBM);
                canvas.drawBitmap(bm, frameToCropTransform, null);
                runInBackground(new Runnable() {
                    @Override
                    public void run() {
                        List<Classifier.Recognition> results = detector.recognizeImage(resizedBM);

                        String lookingForObject = "laptop";
                        for (final Classifier.Recognition result : results) {
                            if (result.getTitle().equals(lookingForObject) && result.getConfidence() > 0.5f) {
                                RectF loc = result.getLocation();
                                float[] center = new float[]{loc.centerX(), loc.centerY()};
                                cropToScreenTransform.mapRect(loc);
                                cropToScreenTransform.mapPoints(center);
                                result.setLocation(loc);
                                if (bestMatch == null || result.getConfidence() > bestMatch.getConfidence()) {
                                    bestMatch = result;
                                    bestMatchCenter[0] = center[0];
                                    bestMatchCenter[1] = center[1];
                                }
                            }
                        }
                        detectingObjects = false;
                    }
                });
            }
        }
        // Submit request for the texture from the current frame.
        gpuDownloadFrameBufferIndex =
                textureReader.submitFrame(backgroundRenderer.getTextureId(), TEXTURE_WIDTH, TEXTURE_HEIGHT);
    }

    private ArrayList<javax.vecmath.Vector3f> getPointCloudCoordinates(Frame frame, float minConfidence) {
        PointCloud pointCloud = frame.acquirePointCloud();
        FloatBuffer pointCloudBuffer = pointCloud.getPoints();
        ArrayList<Vector3f> results = new ArrayList<>(pointCloudBuffer.remaining() / 4);
        Log.d(TAG, "Point cloud buffer remaining: " + pointCloudBuffer.remaining());
        float[] row = new float[4];

        while (pointCloudBuffer.remaining() > 0) {
            pointCloudBuffer.get(row);
            if (row[3] > minConfidence)
                results.add(new Vector3f(row));
        }
        pointCloud.release();
        return results;
    }

    private ArrayList<Vector3f> getPointCloudCoordinates(Frame frame) {
        return getPointCloudCoordinates(frame, 0.0f);
    }

    private Pair<Float, Anchor> getClosestAnchorPair(Pose p) {
        Float leastDistance = null;
        Anchor closestAnchor = null;
        for (Anchor a : session.getAllAnchors()) {
            float curDistance = getPoseSquareDistance(p, a.getPose());
            if (leastDistance == null || leastDistance > curDistance) {
                leastDistance = curDistance;
                closestAnchor = a;
            }
        }
        return new Pair<>(leastDistance, closestAnchor);
    }

    private Anchor getClosestAnchor(Pose p) {
        Pair<Float, Anchor> pair = getClosestAnchorPair(p);
        return pair.second;
    }

    private Float getDistanceToClosestAnchor(Pose p) {
        Pair<Float, Anchor> pair = getClosestAnchorPair(p);
        return pair.first;
    }
}
