using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Entity : MonoBehaviour
{
    [Header("Hierachy Referneces")] [SerializeField]
    private Transform anchorTF;
    
    [Header("Attribute References")]
    [SerializeField] List<ShapeAttribute> shapeAttributes;
    [SerializeField] List<ColorAttribute> colorAttributes;

    public ShapeAttribute shape;
    public ColorAttribute color;
    
    
    
    public void Initialize()
    {
        SetShape();
        SetColor();
    }
    
    public void SetShape()
    {
        shape = shapeAttributes[UnityEngine.Random.Range(0, shapeAttributes.Count)];
        GameObject shapeGO = Instantiate(shape.gameObject, anchorTF.transform);
        shapeGO.SetActive(true);
        shapeGO.transform.localPosition = new Vector3(0, shapeGO.transform.localPosition.y, 0);
        anchorTF.transform.localScale = Vector3.one * UnityEngine.Random.Range(1f, 1.5f);
    }

    public void SetColor()
    {
        color = colorAttributes[UnityEngine.Random.Range(0, colorAttributes.Count)];
        Renderer[] renderers = anchorTF.gameObject.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            Material[] materials = renderer.materials;
            foreach (Material material in materials)
            {
                material.color = color.color;
            }
        }
    }


}

[Serializable] public class EntityAttribute
{
    public string attributeName;
}

[Serializable] public class ColorAttribute : EntityAttribute
{
    public Color color;
}

[Serializable] public class ShapeAttribute : EntityAttribute
{
    public GameObject gameObject;
}