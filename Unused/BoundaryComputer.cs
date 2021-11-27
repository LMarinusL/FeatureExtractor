using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BoundaryComputer : MonoBehaviour
{
    public Vector3[] Vertices;
    public List<Vector2> boundingVectors;
    public Vector2 leftVertex;
    public int startIndex;
    public List<Vector2> boundaryList;
    int secondVert;
    public GameObject dot;




    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.P))
        {
            ComputeBounds();
            InstantiatePoints();
        }
    }
    void ComputeBounds()
    {
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        Vertices = meshGenerator.vertices.ToArray();
        foreach (Vector3 vertex in Vertices)
        {
            if ((meshGenerator.minTerrainHeight + 10) > vertex.y && vertex.y > (meshGenerator.minTerrainHeight + 9.95))
            {
                boundingVectors.Add(new Vector2(vertex.x, vertex.z));
            }
        }
        leftVector(boundingVectors);
        float angle = 0.0f;
        for (int vertex = 0; vertex < boundingVectors.ToArray().Length - 1; vertex++)
        {
            Vector2 v2 = boundingVectors[vertex] - leftVertex;
            float angleNew = Mathf.Atan2(v2.y, v2.x) * Mathf.Rad2Deg;
            if (angleNew > angle)
            {
                angle = angleNew;
                secondVert = vertex;
            }

            List<Vector2> vertofbound = iterateVertices(boundingVectors, startIndex, secondVert);

            // start with right most vertex
            // loop through all vertices, check which has angle closest to 90deg
            // assign that vertex position 2, start again, 
            // with angle between previous 2 vertices
        }
    }
    void leftVector(List<Vector2> list)
    {
        leftVertex = new Vector2(0, 0);
        for (int i = 0; i < list.ToArray().Length - 1; i++)
        {
            if (list[i].x < leftVertex.x)
            {
                startIndex = i;
            }
        }
    }
    List<Vector2> iterateVertices(List<Vector2> list, int firstIndex, int secondIndex)
    {
        int index = 0;
        Vector2 v2 = list[firstIndex] - list[secondIndex];
        float angleNew = Mathf.Atan2(v2.y, v2.x) * Mathf.Rad2Deg;
        for (int i = 0; i < list.ToArray().Length - 1; i++)
        {
            Vector2 v2_2 = list[secondIndex] - list[i];
            float angleNew_2 = Mathf.Atan2(v2_2.y, v2_2.x) * Mathf.Rad2Deg;
            if (angleNew > angleNew_2)
            {
                index = i;
            }
        }
        if (secondIndex != startIndex)
        {
            boundaryList.Add(list[index]);
            iterateVertices(list, secondIndex, index);
        }
        return boundaryList;


    }
    void InstantiatePoints()
    {
        for (int vertId = 0; vertId < boundaryList.ToArray().Length; vertId++) 
        {
            Instantiate(dot, new Vector3(boundaryList[vertId].x, 80, boundaryList[vertId].y) , transform.rotation);
        }
    }
}
