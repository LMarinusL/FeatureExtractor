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

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.J))
        {
            getData();
            getContourVertices(100f, dotone);
            //getContourVertices(110f, dottwo);
            //getContourVertices(90f, dotthree);
            createLine(vectorList);
        }
        if (Input.GetKeyDown(KeyCode.H))
        {
            getData();
            findTriangle();
            createLine(vectorList);
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

    void getContourVertices(float height, GameObject dot)
    {
        Cell currentCell = grid.cells[0];
        Cell previousCell;
        
        float xStep = grid.cells[4 + xSize + 1].x - grid.cells[4].x;
        float zStep = grid.cells[4 + xSize + 1].z - grid.cells[4].z;
        foreach (Cell cell in grid.cells)
        {
            previousCell = currentCell;
            currentCell = cell;

            if (currentCell.y == 0 || previousCell.y == 0 ||
                Mathf.Pow(Mathf.Pow(currentCell.x - previousCell.x, 2) + Mathf.Pow(currentCell.z - previousCell.z, 2), 0.5f) > 2 * xStep) { continue; }
            if ((currentCell.y <= height && previousCell.y > height) ||
                (currentCell.y >= height && previousCell.y < height))
            {
                float ratio = ((currentCell.y - height) / (previousCell.y - currentCell.y));
                contourVertices.Add(new float3(currentCell.x + (xStep * ratio), height,
                    currentCell.z));
            }
        }
        /*  
        Cell sideCell; 
        foreach (Cell cell in grid.cells)
          {
              currentCell = cell;

              if (cell.index < grid.cells.Length - xSize -1)
          {
              sideCell = grid.cells[cell.index + xSize];
              if (currentCell.y == 0 || sideCell.y == 0 ||
              Mathf.Pow(Mathf.Pow(currentCell.x - sideCell.x, 2) + Mathf.Pow(currentCell.z - sideCell.z, 2), 0.5f) > 2 * zStep) { continue; }
              if ((currentCell.y <= height && sideCell.y > height) ||
                  (currentCell.y >= height && sideCell.y < height))
              {
                  float ratio = ((currentCell.y - height) / (sideCell.y - currentCell.y));
                  contourVertices.Add(new float3(currentCell.x, height,
                      currentCell.z + (zStep * ratio)));
              }
          }
      } */
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        vectorList = meshGenerator.float3ToVector3Array(contourVertices.ToArray());
        //foreach (Vector3 point in vectorList)
        //{
        //    Instantiate(dot, point, transform.rotation);
        //}

    }

    void findTriangle()
    {
        // find triangles crossing contour height 
        int vertint = 27105;
        float initHeight = 110f;
        int count = 0;
        List<Face> listFaces = new List<Face>();
        List<Vector3> listFacesV = new List<Vector3>();
        List<Face> outputList;
        outputList = followHeight(grid.cells[vertint].attachedFaces[0], initHeight, listFaces, count);
        Debug.Log(" length contour: " + outputList.ToArray().Length);
        foreach (Face face in outputList)
        {
            listFacesV.Add(new Vector3(face.endVertex.x, face.endVertex.y, face.endVertex.z));
        }
        vectorList = listFacesV.ToArray();
    }

    public List<Face> followHeight(Face start, float contourHeight, List<Face> facesOnHeight, int count)
    {
        facesOnHeight.Add(start);
        count++;
        if (((start.next().ownTriangle.index == facesOnHeight[0].ownTriangle.index || start.previous().ownTriangle.index == facesOnHeight[0].ownTriangle.index) && count > 5 )|| count > 3000)
        {
            return facesOnHeight;
        }
        else
        {
            if ((start.next().endVertex.y >= contourHeight && start.next().startVertex.y < contourHeight)||
                (start.next().endVertex.y < contourHeight && start.next().startVertex.y >= contourHeight))
            {
                return followHeight(start.next().faceTwin, contourHeight, facesOnHeight, count);
            }else {
                if ((start.previous().endVertex.y >= contourHeight && start.previous().startVertex.y < contourHeight )||(
               start.previous().endVertex.y < contourHeight && start.previous().startVertex.y >= contourHeight))
                {
                    return followHeight(start.previous().faceTwin, contourHeight, facesOnHeight, count); // must go to twin
                }
                else
                {
                    return facesOnHeight;
                }
            }
        }
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
