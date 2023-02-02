package com.timegalore.motiondetectionar

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

class CAMShiftDetection(
    targetImage: Mat, initialWindow: Rect,
    erosion_level: Int, erosion_kernel_size: Int, termcrit_count: Int,
    termcrit_eps: Double
) {
    private var g_erosion_level = 10
    private var g_erosion_kernel_size = 4
    private var g_termcrit_count = 10
    private var g_termcrit_eps = 0.01
    private var g_hist = Mat()
    private var g_initialWindow = Rect()
    private val g_firstWindow: Rect

    init {
        g_erosion_level = erosion_level
        g_erosion_kernel_size = erosion_kernel_size
        g_termcrit_count = termcrit_count
        g_termcrit_eps = termcrit_eps
        g_initialWindow = initialWindow
        g_firstWindow = Rect(initialWindow.tl(), initialWindow.br())
        g_hist = getImageHistogram(getHue(targetImage), targetImage.size(), 10, 0f, 180f)
    }

    fun CAMShift(`in`: Mat): RotatedRect {
        val backProjection = getBackProjection(getHue(`in`), g_hist, 0, 180, 1.0)
        val clarifiedBackProjection = clarifyDetectedAreas(backProjection, g_erosion_kernel_size, g_erosion_level)
        validateWindow()
        val rr = doCamShift(clarifiedBackProjection, g_initialWindow, g_termcrit_count, g_termcrit_eps)
        g_initialWindow = rr.boundingRect()
        return rr
    }

    private fun validateWindow() {
        if (g_initialWindow.width < 0 || g_initialWindow.height < 0 || g_initialWindow.width * g_initialWindow.height < 10) {
            Log.d(TAG, "detection window too small - resetting")
            Log.d(TAG, "g first window $g_firstWindow")
            g_initialWindow = Rect(g_firstWindow.tl(), g_firstWindow.br())
        }
    }

    private fun getHue(`in`: Mat): Mat {
        val out = Mat(`in`.size(), CvType.CV_8UC1)
        val hueImage = Mat(`in`.size(), `in`.type())
        Imgproc.cvtColor(`in`, hueImage, Imgproc.COLOR_RGB2HSV)
        Core.extractChannel(hueImage, out, 0)
        return out
    }

    private fun getImageHistogram(huesImage: Mat, size: Size, buckets: Int, minRange: Float, maxRange: Float): Mat {
        val hist = Mat()
        val ranges = MatOfFloat(minRange, maxRange)
        val planes: MutableList<Mat> = ArrayList()
        planes.add(huesImage)
        val chans = MatOfInt(0)
        val histSize = MatOfInt(buckets)
        Imgproc.calcHist(planes, chans, Mat(), hist, histSize, ranges)
        return hist
    }

    private fun clarifyDetectedAreas(`in`: Mat, erosion_kernel_size: Int, erosion_level: Int): Mat {
        val out = Mat(`in`.size(), `in`.type())
        val eroded_kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(erosion_kernel_size.toDouble(), erosion_kernel_size.toDouble()),
            Point((erosion_kernel_size / 2).toDouble(), (erosion_kernel_size / 2).toDouble())
        )
        Imgproc.erode(`in`, out, eroded_kernel, Point(-1.0, -1.0), erosion_level, Core.BORDER_DEFAULT, Scalar(0.0))
        return out
    }

    private fun doCamShift(`in`: Mat, initialWindow: Rect, termcrit_count: Int, termcrit_eps: Double): RotatedRect {
        val termcrit = TermCriteria(TermCriteria.MAX_ITER or TermCriteria.EPS, termcrit_count, termcrit_eps)
        return Video.CamShift(`in`, initialWindow, termcrit)
    }

    private fun getBackProjection(`in`: Mat, histogram: Mat, minRange: Int, maxRange: Int, scale: Double): Mat {
        val images = ArrayList<Mat>()
        images.add(`in`)
        val backproject = Mat(`in`.size(), CvType.CV_8UC1)
        Imgproc.calcBackProject(images, MatOfInt(0), histogram, backproject, MatOfFloat(minRange.toFloat(), maxRange.toFloat()), scale)
        return backproject
    }

    companion object {
        private const val TAG = "CamShiftDetection"
    }
}