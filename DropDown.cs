using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Mathematics;



public class DropDown : MonoBehaviour
{
    public CreateGrid gridImporter;
    public float3[] vertices;

    // Start is called before the first frame update
    void Start()
    {
        GameObject gridcreator = GameObject.Find("GridCreator");
        gridImporter = gridcreator.GetComponent<CreateGrid>();


        vertices = gridImporter.vertices;

        var dropdown = transform.GetComponent<Dropdown>();
        dropdown.options.Clear();

        List<string> items = new List<string>(); 
        items.Add("Choose parameter");
        items.Add("slope");
        items.Add("aspect");
        items.Add("relative slope");
        items.Add("relative aspect");
        items.Add("relative height");
        items.Add("curve");
        items.Add("run-off random");
        items.Add("run-off all cells");
        items.Add("run-off iterate");


        foreach (var item in items)
        {
            dropdown.options.Add(new Dropdown.OptionData() { text = item });

        }
        DropdownItemSelected(dropdown);

        dropdown.onValueChanged.AddListener(delegate { DropdownItemSelected(dropdown); });
    }

    
    void DropdownItemSelected(Dropdown dropdown)
    {
        int index = dropdown.value;
        switch (index)
        {
            case 0:
                break;
            case 1:
                gridImporter.setMeshSlopeColors();
                break;
            case 2:
                gridImporter.setMeshAspectColors();
                break;
            case 3:
                gridImporter.setMeshRelativeSlopeColors();
                break;
            case 4:
                gridImporter.setMeshRelativeAspectColors();
                break;
           case 5:
                gridImporter.setMeshRelativeHeightColors();
                break;
            case 6:
                gridImporter.setMeshCurveColors();
                break;
            case 7:
                List<int> startAt = new List<int>();
                for (int i = 0; i < 1000; i++)
                {
                    startAt.Add(UnityEngine.Random.Range(100, 250000));
                }
                gridImporter.setMeshRunoffColors(startAt.ToArray(), 3000, 20f);
                break;
            case 8:
                int ind = 0;
                int[] array = new int[vertices.Length];
                while (ind < vertices.Length)
                {
                    array[ind] = ind;
                    ind++;
                }
                gridImporter.setMeshRunoffColors(array, 3000, 20f);
                break;
            case 9:
                gridImporter.StartCoroutine(gridImporter.iterate(1000));
                break;
            default:
                // code block
                break;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
