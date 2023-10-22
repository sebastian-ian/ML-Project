using System;
using System.IO;

namespace ML_image_data
{
    class Program
    {
        static void Main(string[] args)
        {
            byte[] bytes = File.ReadAllBytes("train-images (1).idx3-ubyte");
            byte[] newByte = File.ReadAllBytes("train-images (1).idx3-ubyte");
            //byte[] newByte = new byte[32];
            //byte[] compressedFile = File.ReadAllBytes("train-images-updated (1).idx3-ubyte");

            /*
            bytes = new byte[]{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,
                1,0,0,0,0,0,12,5,4,3,0,5,1,0,0,0,0,0,12,88,88,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0};
            File.WriteAllBytes("testDataset.bin", bytes);// copy changes to a new file
            */

            //byte[] bytes = File.ReadAllBytes("file.bin");
            Console.WriteLine(bytes.Length);
            for (int i = 0; i < 100; i++)
            {                          
                //Console.WriteLine(bytes[i] + " + " + compressedFile[i]);
            }
            //Console.WriteLine(bytes[2]);
            int lastByte = 15;
            for (int division = 0; division < 3; division++)
            {
                int newByteIndex = 0;                       
                                       
                //int lastByte = 15 + ((bytes.Length - 16) / 3) * division;
                for (int i = 16 + ((bytes.Length - 16) / 3) * division; i < 16 + ((bytes.Length - 16) / 3) * (division+1); i++)
                {                      
                    if (bytes[i] > 0 || i - lastByte >= 255)
                    {                  
                        if (i - lastByte > 1)
                        {              
                            byte distance = (byte)((i - lastByte) - 1);
                            newByte[newByteIndex] = distance;
                            newByteIndex++;
                            newByte[newByteIndex] = 0;
                            newByteIndex++;
                        }              
                        newByte[newByteIndex] = bytes[i];
                        newByteIndex++;
                        lastByte = i;
                        if (newByteIndex > i)
                        {
                            Console.WriteLine("i>index");
                        }
                    }
                }
                if (division == 2)
                {
                    byte distance = (byte)((bytes.Length - lastByte) - 1);
                    newByte[newByteIndex] = distance;
                    newByteIndex++;
                    newByte[newByteIndex] = 0;
                    newByteIndex++;
                }
                byte[] copyNewByte = new byte[newByteIndex];
                for (int i = 0; i < newByteIndex; i++)
                {
                    copyNewByte[i] = newByte[i];
                }
                Console.WriteLine(newByteIndex);
                //Console.WriteLine(copyNewByte[1]);
                if (division == 0)
                    File.WriteAllBytes("train-images-updated (1).bin", copyNewByte);// copy changes to a new file
                if (division == 1)
                    File.WriteAllBytes("train-images-updated (2).bin", copyNewByte);// copy changes to a new file
                if (division == 2)
                    File.WriteAllBytes("train-images-updated (3).bin", copyNewByte);// copy changes to a new file
                /*
                if (division == 0)
                    File.WriteAllBytes("testDataset1.bin", copyNewByte);// copy changes to a new file
                if (division == 1)
                    File.WriteAllBytes("testDataset2.bin", copyNewByte);// copy changes to a new file
                if (division == 2)
                    File.WriteAllBytes("testDataset3.bin", copyNewByte);// copy changes to a new file
                */
            }
        }
        
    }
}
