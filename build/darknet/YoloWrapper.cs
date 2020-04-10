using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace Darknet
{
#pragma warning disable CA1063 // Implement IDisposable Correctly
    public unsafe class YoloWrapper : IDisposable
#pragma warning restore CA1063 // Implement IDisposable Correctly
    {
        private const string YoloLibraryName = "yolo_cpp_dll.dll";
        private const int MaxObjects = 1000;

#pragma warning disable CA2101 // Specify marshaling for P/Invoke string arguments
        [DllImport(YoloLibraryName, EntryPoint = "init")]
#pragma warning restore CA2101 // Specify marshaling for P/Invoke string arguments
        private static extern int InitializeYolo(string configurationFilename, string weightsFilename, int gpu);

        [DllImport(YoloLibraryName, EntryPoint = "detect_tensor")]
        private static extern int DetectTensor(internal_image_t data, float threshold, ref BboxContainer container);

        [DllImport(YoloLibraryName, EntryPoint = "dispose")]
        private static extern int DisposeYolo();

        [StructLayout(LayoutKind.Sequential)]
        public struct bbox_t
        {
            public UInt32 x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
            public float prob;           // confidence - probability that the object was found correctly
            public UInt32 obj_id;        // class of object - from range [0, classes-1]
            public UInt32 track_id;      // tracking id for video (0 - untracked, 1 - inf - tracked object)
            public UInt32 frames_counter;
            public float x_3d, y_3d, z_3d;  // 3-D coordinates, if there is used 3D-stereo camera
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct BboxContainer
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxObjects)]
            public bbox_t[] candidates;
        }

        public struct image_t
        {
            public Int32 h, w, c;
            public float[] data;
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct internal_image_t
        {
            public Int32 h, w, c;
            public float* data;
        }

        public YoloWrapper(string configurationFilename, string weightsFilename, int gpu)
        {
            InitializeYolo(configurationFilename, weightsFilename, gpu);
        }

#pragma warning disable CA1063 // Implement IDisposable Correctly
        public void Dispose()
#pragma warning restore CA1063 // Implement IDisposable Correctly
        {
            DisposeYolo();
        }

        public bbox_t[] Detect(image_t data, float threshold)
        {
            var container = new BboxContainer();
            fixed (float* pData = data.data)
            {
                var internalData = new internal_image_t { w = data.w, h = data.h, c = data.c, data = pData };
                var count = DetectTensor(internalData, threshold, ref container);
                return container.candidates.Take(count).ToArray();
            }
        }
    }
}
