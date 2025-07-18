package com.appliedrec.verid3.facedetection.retinaface

/**
 * NNAPI options
 *
 * @property flag Bitwise flag to use with NNAPI
 */
enum class NnapiOptions(val flag: Int) {
    /**
     * Use 16-bit floating point precision
     */
    USE_FP16(0x001),

    /**
     * Disable CPU inference
     */
    CPU_DISABLED(0x004);

    companion object {
        /**
         * Create a set of NnapiOptions from a bitwise flag
         *
         * @param flags Bitwise flag
         * @return Set of NnapiOptions
         */
        @JvmStatic
        fun fromFlags(flags: Int): Set<NnapiOptions> {
            return entries.filter { (flags and it.flag) != 0 }.toSet()
        }
    }
}

/**
 * Convert a set of NnapiOptions to a bitwise flag
 *
 * @return Bitwise flag
 */
fun Set<NnapiOptions>.toFlags(): Int = fold(0) { acc, opt -> acc or opt.flag }