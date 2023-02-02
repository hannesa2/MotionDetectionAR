package com.timegalore.motiondetectionar

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.*
import android.view.View.OnTouchListener
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.JavaCameraView
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgproc.Imgproc
import rajawali.RajawaliActivity
import java.io.File
import java.text.DecimalFormat


class MotionDetectionActivity : RajawaliActivity(), CvCameraViewListener2, OnTouchListener, SensorEventListener {
    // Accelerometer
    var mSensor: SensorManager? = null
    private var mRenderer: OpenGLRenderer? = null
    private var mItemPreviewCaptureImage: MenuItem? = null
    private var mItemPreviewSampleImage: MenuItem? = null
    private var mItemCamShift: MenuItem? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var loadedImage: Mat? = null
    private var mRgba: Mat? = null
    private val showThumbs = true
    private var showEllipse = true
    private var csd: CAMShiftDetection? = null
    private var mfd: MotionFlowDetection? = null

    // Accelerometer
    private fun initOpenCV() {
        mOpenCvCameraView!!.enableView()
        Log.d(TAG, "loading file")

        // https://stackoverflow.com/q/18103994/1079990
        val image= BitmapFactory.decodeResource(resources,R.drawable.red)
        loadedImage = Mat(image.height, image.width, CvType.CV_8U, Scalar(4.0))
        val myBitmap32: Bitmap = image.copy(Bitmap.Config.ARGB_8888, true)
        Utils.bitmapToMat(myBitmap32, loadedImage)

//        loadedImage = imread(resources.getDrawable(R.drawable.red).toString())
//        loadedImage = imread("/storage/emulated/0/red.jpg")
//        loadedImage = loadImageFromFile("red.jpg")
        val initialWindow = Rect(
            loadedImage!!.width() / 3,
            loadedImage!!.height() / 3, loadedImage!!.width() * 2 / 3,
            loadedImage!!.height() * 2 / 3
        )
        csd = CAMShiftDetection(
            loadedImage!!, initialWindow, 10, 4,
            10, 0.01
        )
    }

    /**
     * Called when the activity is first created.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        mOpenCvCameraView = JavaCameraView(this, -1)
        mOpenCvCameraView!!.setCvCameraViewListener(this)
        mLayout.addView(mOpenCvCameraView)
        mSurfaceView.setZOrderMediaOverlay(true)
        setGLBackgroundTransparent(true)
        mRenderer = OpenGLRenderer(this)
        mRenderer!!.surfaceView = mSurfaceView
        super.setRenderer(mRenderer)
        mRenderer!!.setCameraPosition(0.0, 0.0, 20.0)

        initOpenCV()
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    public override fun onResume() {
        super.onResume()
        mOpenCvCameraView!!.setOnTouchListener(this)
        initialiseSensor()
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        Log.i(TAG, "called onCreateOptionsMenu")
        mItemPreviewCaptureImage = menu.add("Capture Image")
        mItemPreviewSampleImage = menu.add("Sample Image")
        mItemCamShift = menu.add("Cam Shift")
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        Log.d(TAG, "called onOptionsItemSelected; selected item: $item")
        if (item === mItemPreviewCaptureImage) viewMode = VIEW_MODE_CAPTUREIMAGE else if (item === mItemPreviewSampleImage) viewMode =
            VIEW_MODE_SHOWIMAGE else if (item === mItemCamShift) viewMode = VIEW_MODE_CAMSHIFT
        return true
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat()
    }

    override fun onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mRgba != null) mRgba!!.release()
        mRgba = null
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val df = DecimalFormat("#.##")
        mRgba = inputFrame.rgba()
        when (viewMode) {
            VIEW_MODE_CAPTUREIMAGE -> {
                val w = mRgba!!.width()
                val h = mRgba!!.height()
                Imgproc.rectangle(
                    mRgba, Point((w * 1 / 3).toDouble(), (h * 1 / 3).toDouble()), Point(
                        (
                                w * 2 / 3).toDouble(), (h * 2 / 3).toDouble()
                    ), Scalar(255.0, 0.0, 0.0, 255.0)
                )
            }
            VIEW_MODE_SHOWIMAGE -> Imgproc.resize(loadedImage, mRgba, mRgba!!.size())
            VIEW_MODE_CAMSHIFT -> {
                val rr = csd!!.CAMShift(mRgba!!)
                if (showEllipse) Imgproc.ellipse(mRgba, rr, Scalar(255.0, 255.0, 0.0), 5)
                if (mfd == null) mfd = MotionFlowDetection(mRgba!!.size())
                val leftRightRot = mfd!!.motionFlowDetection(mRgba!!)
                Imgproc.putText(
                    mRgba, "x: " + rr.center.x.toInt() + " x: " + mSensorX
                            + " y: " + mSensorY + " z: " + mSensorZ + " r: "
                            + leftRightRot, Point(0.0, 30.0),
                    Imgproc.FONT_HERSHEY_COMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2
                )
                if (mRenderer!!.isReady) augmentImage(
                    mRgba, rr, mSensorX, mSensorY, mSensorZ,
                    leftRightRot
                )
            }
        }
        return mRgba!!
    }

    private fun augmentImage(mRgba: Mat?, rr: RotatedRect, mSensorX: Int, mSensorY: Int, mSensorZ: Int, leftRightRot: Int) {

        // X is to right of device when facing it
        // Y is though top of device when facing it
        // Z is coming straight out of the screen when facing it

        // draw line through the centre of the object along the z axis
        // with phone vertical the in line would be straight up and down
        // and rotation left/right would cause line angle to change

        // Front/Back rotate is simply Z
        // rotation clock/anticlockwise is slight harder
        // but doesn't involve Z
        // in landscape: X is 10 and Y is 0
        // in portrait X is 0 and Y is 10
        // in upside down landscape: X is -10 and Y is 0
        // in upside down portrait: X is 0 and Y is -10
        // so angle of rotation where normal portrait is say 0 degrees
        // is: ATAN2(Y,-X)+PI/2
        // left/right movement is Y but depends on the clock
        // rotation so need to factor this in as Y * Cos (angle)
        val centre = rr.center
        val lrTiltAngleInRadians = Math.atan2(mSensorY.toDouble(), mSensorX.toDouble())
        var fbTiltAngleInRadians = Math.PI / 2 * Math.sin(mSensorZ / 10.0)

        // due to limitations on sensor information, the phone cannot
        // distinguish between, say, a landscape view from the top with
        // an inverted landscape view from the bottom - ie. the sensors
        // will show all the same readings in both case
        // the trick therefore is to use sign of the Z setting to flip
        // the object when X becomes negative.
        if (mSensorX < 0 && mSensorZ > 0) {
            // fbTiltAngleInRadians += Math.PI;
            // lrTiltAngleInRadians = -lrTiltAngleInRadians;
            fbTiltAngleInRadians += Math.PI
            mRenderer!!.setSpin(0.0)
        }
        val df = DecimalFormat("#.##")
        Log.d(
            TAG,
            "x:" + mSensorX + " y:" + mSensorY + " z:" + mSensorZ + " rot:" + df.format(lrTiltAngleInRadians) + " fb:" + df.format(
                fbTiltAngleInRadians
            ) + " lr:" + df.format(leftRightRot.toLong())
        )
        setPosition(centre.x, centre.y)
        mRenderer!!.setCamLRTilt(-lrTiltAngleInRadians)
        mRenderer!!.setCamFBTilt(-fbTiltAngleInRadians)
        val cs = Math.sqrt(rr.boundingRect().area())
        mRenderer!!.setCubeSize(2 * cs / 480) // 0.6 for pegasus
    }

    fun printMatDetails(name: String, m: Mat) {
        Log.d(
            TAG,
            name + " - " + "c:" + m.channels() + ",cols:" + m.cols() + ",dep:" + m.depth() + ",rows:" + m.rows() + ",type:" + m.type() + ",w:" + m.width() + ",h:" + m.height()
        )
    }

    override fun onTouch(v: View, event: MotionEvent): Boolean {
        Log.d(TAG, "got touch " + event.action)
        val x = event.x
        val y = event.y
        Log.d(TAG, "x=$x,y=$y")

        // setPosition(x, y);
        Log.d(TAG, "object pos: " + mRenderer!!.get3DObjectPosition().toString())
        if (viewMode == VIEW_MODE_CAPTUREIMAGE) {
            val w = mRgba!!.width()
            val h = mRgba!!.height()

            // +1 to x,y to avoid cutting the red line of the viewfinder box
            val roi = Rect(Point((w * 1 / 3 + 1).toDouble(), (h * 1 / 3 + 1).toDouble()), Point((w * 2 / 3).toDouble(), (h * 2 / 3).toDouble()))
            val viewFinder = mRgba!!.submat(roi)
            Imgproc.resize(viewFinder, loadedImage, loadedImage!!.size())
            val initialWindow = Rect(
                loadedImage!!.width() / 3,
                loadedImage!!.height() / 3, loadedImage!!.width() * 2 / 3,
                loadedImage!!.height() * 2 / 3
            )
            csd = CAMShiftDetection(loadedImage!!, initialWindow, 10, 4, 10, 0.01)
        }
        if (viewMode == VIEW_MODE_CAMSHIFT) {
            showEllipse = !showEllipse
        }
        return false
    }

    private fun setPosition(x: Double, y: Double) {
        val cD = mRenderer!!.currentCamera.z
        val yVP = mRenderer!!.viewportHeight.toFloat()
        val xVP = mRenderer!!.viewportWidth.toFloat()

        // =(K16-xVP/2)* (cD/xVP)
        // =(K17-yVP/2)* (cD/2/yVP)
        val sx = 0.7
        val sy = 1.3
        val obx = (x - xVP / 2) * (cD / sx / xVP)
        val oby = (yVP / 2 - y) * (cD / sy / yVP)
        mRenderer!!.set3DObjectPosition(obx, oby, 0.0)
    }

    private fun loadImageFromFile(fileName: String): Mat? {
        var rgbLoadedImage: Mat? = null
        val root = Environment.getExternalStorageDirectory()
        val file = File(root, fileName)

        // this should be in BGR format according to the documentation.
        Log.w("LoadFile", file.absolutePath)
        val image = imread(file.absolutePath)
        if (image.width() > 0) {
            rgbLoadedImage = Mat(image.size(), image.type())
            Imgproc.cvtColor(image, rgbLoadedImage, Imgproc.COLOR_BGR2RGB)
            Log.d(TAG, "loadedImage: channelss: " + image.channels() + ", (" + image.width() + ", " + image.height() + ")")
            image.release()
        }
        return rgbLoadedImage
    }

    fun writeImageToFile(image: Mat, filename: String?) {
        val root = Environment.getExternalStorageDirectory()
        val file = File(root, filename)
        Imgcodecs.imwrite(file.absolutePath, image)
        Log.d(TAG, "writing: " + file.absolutePath + " (" + image.width() + ", " + image.height() + ")")
    }

    private fun initialiseSensor() {
        if (mSensor == null) mSensor = getSystemService(SENSOR_SERVICE) as SensorManager
        mSensor!!.registerListener(
            this,
            mSensor!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            SensorManager.SENSOR_DELAY_GAME
        )
    }

    override fun onAccuracyChanged(arg0: Sensor, arg1: Int) {}
    override fun onSensorChanged(event: SensorEvent) {
        val vals = event.values
        mSensorX = vals[0].toInt()
        mSensorY = vals[1].toInt()
        mSensorZ = vals[2].toInt()
    }

    companion object {
        const val VIEW_MODE_CAPTUREIMAGE = 2
        const val VIEW_MODE_SHOWIMAGE = 3
        const val VIEW_MODE_CAMSHIFT = 8
        private const val TAG = "ImageManipulation"
        var viewMode = VIEW_MODE_CAMSHIFT
        private var mSensorX = 0
        private var mSensorY = 0
        private var mSensorZ = 0

        init {
            System.loadLibrary("opencv_java4")
        }
    }

}
