package com.timegalore.motiondetectionar

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs

class MotionFlowDetection(var imageSize: Size) {
    var initial: MatOfPoint? = null
    var status: MatOfByte? = null
    var err: MatOfFloat? = null
    var prevPts: MatOfPoint2f? = null
    var nextPts: MatOfPoint2f? = null
    var maxCorners = 0
    private var mGray1: Mat? = null
    private var mGray2: Mat? = null

    fun motionFlowDetection(prevImage: Mat, nextImage: Mat?): Point {
        var direction: Point?
        setOpticalFlowParameters(prevImage)
        val resultImage = prevImage.clone()
        Imgproc.cvtColor(prevImage, mGray1, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.cvtColor(nextImage, mGray2, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.goodFeaturesToTrack(mGray1, initial, 1000, 0.01, 5.0)
        initial!!.convertTo(prevPts, CvType.CV_32FC2)
        Video.calcOpticalFlowPyrLK(
            mGray1, mGray2, prevPts, nextPts, status,
            err
        )
        val pointp = prevPts!!.toArray()
        val pointn = nextPts!!.toArray()
        markPointsOnImage(resultImage, pointp, pointn)
        direction = getAverageDirection(pointp, pointn)
        return direction
    }

    private fun getAverageDirection(pointp: Array<Point>, pointn: Array<Point>): Point {
        val p = Point()
        val nosOfPoints = pointp.size
        for (i in 0 until nosOfPoints) {
            p.x += pointp[i].x - pointn[i].x
            p.y += pointp[i].y - pointn[i].y
        }
        p.x = p.x / nosOfPoints
        p.y = p.y / nosOfPoints
        return p
    }

    private fun markPointsOnImage(resultImage: Mat, pointp: Array<Point>, pointn: Array<Point>) {
        for (i in pointp.indices) {
            val distanceX = abs(pointn[i].x - pointp[i].x).toInt()
            val distanceY = abs(pointn[i].y - pointp[i].y).toInt()
            Imgproc.circle(resultImage, pointn[i], 10, Scalar(255.0, 0.0, 0.0, 255.0))
        }
    }

    fun motionFlowDetection(image: Mat): Int {
        setOpticalFlowParameters(image)
        mGray1 = mGray2!!.clone()
        Imgproc.cvtColor(image, mGray2, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.goodFeaturesToTrack(mGray1, initial, maxCorners, 0.01, 5.0)
        initial!!.convertTo(prevPts, CvType.CV_32FC2)
        Video.calcOpticalFlowPyrLK(mGray1, mGray2, prevPts, nextPts, status, err)
        val pointp = prevPts!!.toArray()
        val pointn = nextPts!!.toArray()
        val dir = calculateVelocityFromMotionFlow(pointn, pointp)
        return removeOutliersAndNoise(dir)
    }

    private fun removeOutliersAndNoise(vGiven: Int): Int {
        var v = vGiven
        if (v < 0 && v > -10)
            v = 0
        if (v in 1..9)
            v = 0
        if (v < 0 && v < -80)
            v = -80
        if (v > 0 && v > 80)
            v = 80
        return v
    }

    private fun calculateVelocityFromMotionFlow(pointn: Array<Point>, pointp: Array<Point>): Int {

        // find the average difference from all the analysed points. The sign of
        // the
        // average gives you the direction
        val points = pointn.size
        var total = 0
        for (i in 0 until points) {
            total += (pointn[i].x - pointp[i].x).toInt()
        }
        return total / points
    }

    private fun setOpticalFlowParameters(image: Mat) {
        if (mGray1 == null)
            mGray1 = Mat(imageSize, CvType.CV_8UC1)
        if (mGray2 == null) {
            mGray2 = Mat(imageSize, CvType.CV_8UC1)
            Imgproc.cvtColor(image, mGray2, Imgproc.COLOR_RGBA2GRAY)
        }
        initial = MatOfPoint()
        status = MatOfByte()
        err = MatOfFloat()
        prevPts = MatOfPoint2f()
        nextPts = MatOfPoint2f()
        maxCorners = 10
    }
}