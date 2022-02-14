using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System.Linq;
using System;
using Unity.Mathematics;
public class ContourGenerator : MonoBehaviour
{

    public Grid grid;
    public int xSize;
    public int zSize;
    public GameObject dotone;
    public GameObject dottwo;
    public GameObject dotthree;
    public GameObject lineObject;
    public Vector3[] vectorList;
    public List<float3> contourVertices;
    public LineRenderer line;
    public Material material;
    public Vector3[] orderedArray;
    public List<Face> faces;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.H))
        {
            getData();
            for (int i = 0; i < 100; i++)
            {
                createLine(contourSegment(90f));
                createLine(contourSegment(100f));
                createLine(contourSegment(110f));
                createLine(contourSegment(80f));
                createLine(contourSegment(70f));
                createLine(contourSegment(60f));
            }
        }

    }

    void getData()
    {
        GameObject gridcreator = GameObject.Find("GridCreator");
        CreateGrid gridImporter = gridcreator.GetComponent<CreateGrid>();

        grid = gridImporter.grid;
        xSize = gridImporter.xSize;
        zSize = gridImporter.zSize;

        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
    }

    int findFace(float height)
    {
        int toCheck = 0;
        bool found = false;
        int iter = 0;
        while (found == false && iter<10000)
        {

            toCheck = UnityEngine.Random.Range(0, grid.cells.Length);
            float height1 = grid.cells[toCheck].attachedFaces[0].startVertex.y;
            float height2 = grid.cells[toCheck].attachedFaces[0].endVertex.y;
            if ((height1 >= height && height2 < height) || (height1 < height && height2 >= height))
            {
                found = true;
                Debug.Log("vertex: " + grid.cells[toCheck].x + " " + grid.cells[toCheck].y + " " + grid.cells[toCheck].z);
            }
            iter++;
        }
        return toCheck;
    }

        Vector3[] contourSegment(float height)
    {
        int vert = findFace(height);
        int count = 0;
        List<Face> listFaces = new List<Face>();
        List<Vector3> listVertices;
        List<Face> outputList;
        int maxCount = 1000;
        outputList = followHeight(grid.cells[vert].attachedFaces[0], height, count, maxCount, listFaces);
        faces.AddRange(outputList);
        listVertices = faceToVertex(outputList, height);
        Debug.Log(" length contour: " + outputList.ToArray().Length);
        return listVertices.ToArray();
    }

    public List<Face> followHeight(Face start, float contourHeight, int count, int maxCount, List<Face> facesOnHeight)
    {
        facesOnHeight.Add(start);
        count++;
        try
        {
            if (((start.next().ownTriangle.index == facesOnHeight[0].ownTriangle.index || start.previous().ownTriangle.index == facesOnHeight[0].ownTriangle.index) && count > 5) || count > maxCount)
            {
                return facesOnHeight;
            }
            else
            {
                if ((start.next().endVertex.y >= contourHeight && start.next().startVertex.y < contourHeight) ||
                    (start.next().endVertex.y < contourHeight && start.next().startVertex.y >= contourHeight))
                {
                    return followHeight(start.next().faceTwin, contourHeight, count, maxCount, facesOnHeight);
                }
                else
                {
                    if ((start.previous().endVertex.y >= contourHeight && start.previous().startVertex.y < contourHeight) || (
                   start.previous().endVertex.y < contourHeight && start.previous().startVertex.y >= contourHeight))
                    {
                        return followHeight(start.previous().faceTwin, contourHeight, count, maxCount, facesOnHeight);
                    }
                    else
                    {
                        return facesOnHeight;
                    }
                }
            }
        }
        catch
        {
            Debug.Log(" an error ocurred here");
            facesOnHeight.RemoveAt(facesOnHeight.Count - 1);
            return facesOnHeight;
        }
    }

    List<Vector3> faceToVertex(List<Face> facesOnHeight, float height)
    {
        float xStep = grid.cells[4 + xSize + 1].x - grid.cells[4].x;
        float zStep = grid.cells[4 + xSize + 1].z - grid.cells[4].z;
        List<Vector3> outputList = new List<Vector3>();
        int i = 0;
        foreach(Face face in facesOnHeight)
        {
            if (face.startVertex.y != 0 && face.endVertex.y != 0)
            {
                float ratio = ((face.startVertex.y - height) / (face.endVertex.y - face.startVertex.y));
                outputList.Add( new Vector3(face.startVertex.x + (xStep * ratio), height,
                    face.startVertex.z + (zStep * ratio)));
                i++;
            }
        }
        return outputList;

    }

    void createLine(Vector3[] vectorList)
    {
        lineObject = new GameObject("Line");
        line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 2f;
        line.endWidth = 2f;
        line.positionCount = vectorList.Length;
        line.material = material;
        for (int i=0; i < vectorList.Length; i++)
        {
            line.SetPosition(i, vectorList[i]);
        }
    }
}
