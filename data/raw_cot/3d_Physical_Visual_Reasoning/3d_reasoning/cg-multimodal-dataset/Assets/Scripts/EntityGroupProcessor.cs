using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EntityGroupProcessor 
{
    List<Entity> allEntities = new List<Entity>();
    
    public void AddEntity(Entity entity)
    {
        allEntities.Add(entity);
    }

    public void Clear()
    {
        allEntities.Clear();
    }

    public string ProcessEntities()
    {
        // Select a random entitiy
        Entity randomEntity = allEntities[UnityEngine.Random.Range(0, allEntities.Count)];

        // Select a random attribute
        int attributeType = Random.Range(0, 2); // 0 - Shape, 1 - Color
        string attributeName = "";

        if (attributeType == 0)
        {
            attributeName = randomEntity.shape.attributeName;
        }else if (attributeType == 1)
        {
            attributeName = randomEntity.color.attributeName;
        }
        
        // Hide all entities with the same attribute
        int hideAmount = 0;
        
        foreach (var VARIABLE in allEntities)
        {
            if (attributeType == 0)
            {
                if(VARIABLE.shape.attributeName == attributeName)
                {
                    VARIABLE.gameObject.SetActive(false);
                    hideAmount++;
                }
            }else if (attributeType == 1)
            {
                if(VARIABLE.color.attributeName == attributeName)
                {
                    VARIABLE.gameObject.SetActive(false);
                    hideAmount++;
                }
            }
        }
        
        return "Removed all " + attributeName + " objects. \n Press [SPACE] to regenerate scene.";
    }
}
