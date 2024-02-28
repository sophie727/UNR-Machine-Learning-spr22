#include <bits/stdc++.h>

using namespace std;



/*
 * Complete the 'findLastOctal' function below.
 *
 * The function is expected to return an INTEGER.
 * The function accepts STRING s as parameter.
 */

string convertBaseFromInt (int n, int b) {
    string converted = "";
    while (n > 0) {
        converted.push_back(n % b);
        n /= b;
    } // this results in the reverse of the number we want
    reverse(converted.begin(), converted.end());
    return converted;
}

int convertIntFromBase (string n, int b) {
    //n is the number in base b; we want to convert it to base 10.
    int ans = 0;
    
    reverse(n.begin(), n.end());
    int pow = 1;
    for (int i = 0; i < n.length(); i++) {
        int dig = int(n[i]) - 48;
        ans += dig * pow;
        pow *= b;
    }
    
    return ans;
}

string convertToBaseASCII(string s, int b) {
    string converted = "";
    for (int i = 0; i < s.length(); i++) {
        int ascii = int(s[i]);
        string convertedAsc = convertBaseFromInt(ascii, b);
        converted += convertedAsc;
    }
    return converted;
}

int searchForNum(string& s, string n, bool front) {
    // the idea is that we want to search through the string s for a string n. If bool front == true, we search from the start; if not, we search from the end. It returns the index of where the string n starts, or if we can't find it, return -1.
    if(s.length() < n.length())
        return -1;
        
    for(int i = 0; i < s.length() - n.length() + 1; i++) {
        int index; 
        if (front == true)
            index = i;
        else
            index = s.length() - n.length() - i;
    
        bool matches = true;
        for(int j = 0; j < n.length(); j++) {
            if (s[index + j] != n[j])
                matches = false;
        }
        
        if (matches == true)
            return index;
    }
    return -1;
}

int findLastOctal(string s) {
    s = convertToBaseASCII(s, 2);
    
    bool found = true;
    int n = 1; //we'll be removing this number from the string
    while (found == true) {
        string remove = convertBaseFromInt(n, 2);
        int index = searchForNum(s, remove, true);
        if (index == -1) {
            found = false;
            break;
        }
        s.erase(index, remove.length());
        
        index = searchForNum(s, remove, false);
        if (index == -1) {
            found = false;
            break;
        }
        s.erase(index, remove.length());
    }
    
    int temp = convertIntFromBase(s, 2);
    s = convertBaseFromInt(temp, 8);
    
    while (found == true) {
        string remove = convertBaseFromInt(n, 8);
        int index = searchForNum(s, remove, true);
        if (index == -1) {
            found = false;
            break;
        }
        s.erase(index, remove.length());
        
        index = searchForNum(s, remove, false);
        if (index == -1) {
            found = false;
            break;
        }
        s.erase(index, remove.length());
    }
    
    int ans = convertIntFromBase(s, 8);
    
    return ans;
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    string s;
    getline(cin, s);

    int result = findLastOctal(s);

    fout << result << "\n";

    fout.close();

    return 0;
}

