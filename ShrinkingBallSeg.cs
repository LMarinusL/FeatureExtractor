using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;


    public class ShrinkingBallSeg : MonoBehaviour
    {
        public Vector3[] vertices;
        public Vector3[] normals;
        public float initialRadius = 200.0f;
        public List<Vector3> filteredList;
        public List<Vector3> MedialBallCenters;
        public List<float> MedialBallRadii;
        public GameObject dotred;
        public GameObject dotblue;
        MeshComponent meshComp;
        public MATList list;

        void Start()
        {

        }

        void Update()
        {
            if (Input.GetKey(KeyCode.I))
            {
                getMesh();
                meshComp = new MeshComponent(vertices);
                iterateVertices();
                InstantiatePoints();
            }
        }

        public void getMesh()
        {
            GameObject terrain = GameObject.Find("TerrainLoader");
            MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
            Mesh mesh = meshGenerator.mesh;
            vertices = mesh.vertices;
            normals = mesh.normals;
        }

        bool checkRadius(int vertexIndex, float radius)
        {
            Vector3 medialBallCenter = vertices[vertexIndex] + normals[vertexIndex] * radius;
            List<Vector3> list = meshComp.checkSegment(medialBallCenter, radius);
            foreach (Vector3 vertex in list)
            {
                if (Vector3.Distance(vertex, medialBallCenter) < radius)
                {
                    return false;
                }
            }
            return true;
        }

        void getMedialBallCenter(int vertexIndex)
        {
            bool empty = false;
            float radius = initialRadius;
            while (empty == false)
            {
                radius -= 3.0f;
                empty = checkRadius(vertexIndex, radius);
                if (radius < 10f)
                {
                    return;
                }
            }
            MedialBallCenters.Add((vertices[vertexIndex] + normals[vertexIndex] * radius));
            MedialBallRadii.Add(radius);
        }

        void iterateVertices()
        {
            for (int i = 0; i < vertices.Length; i++)
            {
                if (vertices[i].y != 0f && normals[i].y > 0.4f)
                {
                    getMedialBallCenter(i);
                }
            }
        }
        void filterOnRadius()
        {

        }
        void InstantiatePoints()
        {
            list = new MATList(MedialBallCenters.ToArray());
            list.setScores();
            //WriteString(MedialBallCenters);
            for (int vertId = 0; vertId < MedialBallCenters.ToArray().Length; vertId++)
            {
                Instantiate(dotblue, list.getLoc3D(vertId), transform.rotation);
                Instantiate(dotred, list.getLoc2D(vertId, 300f), transform.rotation);
            }
        }

        /*public static void WriteString(List<Vector3> list)
        {
            string path = "Assets/Output/test.txt";
            StreamWriter writer = new StreamWriter(path, false);
            foreach (Vector3 vector in list)
            {
                writer.WriteLine(vector.x + " " + vector.y + " " + vector.z);
            }
            writer.Close();
        }*/
    }

    public class MATList : Component
    {
        public Vector3[] OriginalMATList;
        public List<MATBall> NewMATList = new List<MATBall>();


        public MATList(Vector3[] originalMATList) // constructor
        {
            OriginalMATList = originalMATList;
            foreach (Vector3 ball in OriginalMATList)
            {
                NewMATList.Add(new MATBall(ball));
            }
        }
        public void setScores()
        {
            MeshComponent matComp = new MeshComponent(OriginalMATList);
            float radiusForScore = 100f;
            for (int num = 0; num < OriginalMATList.ToArray().Length; num++)
            {
                List<Vector3> listToCheck = matComp.checkSegment(OriginalMATList[num], radiusForScore);
                int score = 0;
                foreach (Vector3 vertex in listToCheck)
                {
                    if (Vector3.Distance(vertex, OriginalMATList[num]) < radiusForScore)
                    {
                        score++;
                    }
                }
                NewMATList[num].Score = score;
            }
        }
        public Vector3 getLoc3D(int index)
        {
            return NewMATList[index].Loc;
        }
        public Vector3 getLoc2D(int index, float yloc)
        {
            return new Vector3(NewMATList[index].Loc.x, yloc, NewMATList[index].Loc.z);
        }
        public MATBall getBall(int index)
        {
            return NewMATList[index];
        }

    }

    public class MATBall : Component
    {
        public Vector3 Loc;
        public int Score;

        public MATBall(Vector3 ballLoc) // constructor
        {
            Loc = ballLoc;
        }
    }

public class MATBranch : Component
{
    public Vector3 Head;
    public Vector3 Tail;

    public MATBranch(Vector3 head, Vector3 tail)
    {
        Head = head;
        Tail = tail;
    }
}



