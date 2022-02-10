using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;


public class ShrinkingBallSeg : MonoBehaviour
    {
        public float3[] vertices;
        public float3[] normals;
        public float initialRadius = 100.0f;
        public List<float3> MedialBallCenters;
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
            if (Input.GetKeyDown(KeyCode.I))
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
            vertices = meshGenerator.vertices;
            mesh.RecalculateNormals();
            normals = meshGenerator.normals;
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

    bool checkRadius(int vertexIndex, float radius)
    {
        float3 medialBallCenter = vertices[vertexIndex] + normals[vertexIndex] * radius;
        float3[] list = meshComp.checkSegment(medialBallCenter, radius);
        foreach (float3 vertex in list)
        {
            if (Distance(vertex, medialBallCenter) < radius)
            {
                return false;
            }
        }
        return true;
    }

    void InstantiatePoints()
        {
            list = new MATList(MedialBallCenters.ToArray());
            list.setScores();
            for (int vertId = 0; vertId < MedialBallCenters.ToArray().Length; vertId++)
            {
                Instantiate(dotblue, list.getLoc3D(vertId), transform.rotation);
                //Instantiate(dotred, list.getLoc2D(vertId, 300f), transform.rotation);
            }
        }
    
    float Distance(float3 point1, float3 point2)
    {
        return Mathf.Pow(Mathf.Pow(point1.x - point2.x, 2f) + Mathf.Pow(point1.y - point2.y, 2f) + Mathf.Pow(point1.z - point2.z, 2f), 0.5f);
    }

    }

    public class MATList : Component
    {
        public float3[] OriginalMATList;
        public MATBall[] NewMATList;


        public MATList(float3[] originalMATList) // constructor
        {
            OriginalMATList = originalMATList;
            NewMATList = new MATBall[originalMATList.Length];
            int i = 0;
            foreach (float3 ball in OriginalMATList)
            {
                NewMATList[i] = new MATBall(ball);
                i++;
            }
        }
        public void setScores()
        {
            MeshComponent matComp = new MeshComponent(OriginalMATList);
            float radiusForScore = 100f;
            for (int num = 0; num < OriginalMATList.Length; num++)
            {
            float3[] listToCheck = matComp.checkSegment(OriginalMATList[num], radiusForScore);
                int score = 0;
                foreach (float3 vertex in listToCheck)
                {
                    if (Distance(vertex, OriginalMATList[num]) < radiusForScore)
                    {
                        score++;
                    }
                }
                NewMATList[num].Score = score;
            }
        }
        public float3 getLoc3D(int index)
        {
            return NewMATList[index].Loc;
        }
        public float3 getLoc2D(int index, float yloc)
        {
            return new float3(NewMATList[index].Loc.x, yloc, NewMATList[index].Loc.z);
        }
        public MATBall getBall(int index)
        {
            return NewMATList[index];
        }

    float Distance(float3 point1, float3 point2)
    {
        return Mathf.Pow(Mathf.Pow(point1.x - point2.x, 2f) + Mathf.Pow(point1.y - point2.y, 2f) + Mathf.Pow(point1.z - point2.z, 2f), 0.5f);
    }

}

    public class MATBall : Component
    {
        public float3 Loc;
        public int Score;

        public MATBall(float3 ballLoc) // constructor
        {
            Loc = ballLoc;
        }
    }

public class MATBranch : Component
{
    public float3 Head;
    public float3 Tail;

    public MATBranch(float3 head, float3 tail)
    {
        Head = head;
        Tail = tail;
    }
}



